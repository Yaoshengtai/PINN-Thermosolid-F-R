from math import pi as PI
import torch
from torch import nn, optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import ast

from PINN.diff import diff
from PINN.networks import FCNN
from PINN.conditions import BoundaryCondition
from PINN.generators import generator_2dspatial_rectangle, generator_2dspatial_segment
from PINN.solvers import SingleNetworkApproximator2DSpatial
from PINN.monitors import Monitor2DSpatial
from PINN.solvers import _solve_2dspatial

r1=0.135 #内径
r2=0.1695  #外径
h1=0.02  #高

E= 420*10**-3 #杨氏模量 um $ MPa
mu =0.14 #泊松比
G=E/2/(1+mu)  #剪切模量
alpha=4*10**-6 #线膨胀系数
beta=alpha *E /(1-2*mu) *10**6#热应力系数
maxf=10 #最高温度

parser = argparse.ArgumentParser(description='PyTorch Deep Learning Training Force Forward')

# 添加命令行参数
parser.add_argument('--lr', type=float, default=0.00002, help='学习率')
parser.add_argument('--batch_size', type=int, default=1024, help='批量大小')
parser.add_argument('--epochs', type=int, default=1000000, help='训练轮数')
parser.add_argument('--gpu', type=bool , default=False ,help='使用GPU进行训练')
parser.add_argument('--train_rec_size', type=int , default=256 ,help='矩形区域内生成的点,512*512')
parser.add_argument('--train_bound_size', type=int , default=64 ,help='边界上生成的点数')
parser.add_argument('--train_gen_random', type=bool , default=True ,help='训练生成点是否随机')
parser.add_argument('--valid_gen_random', type=bool , default=True ,help='验证生成点是否随机')
parser.add_argument('--weight_up', type=int , default=20 ,help='上边界权重')
parser.add_argument('--weight_left', type=int , default=10 ,help='左边界权重')
parser.add_argument('--weight_right', type=int , default=5 ,help='右边界权重')
parser.add_argument('--weight_bottom', type=int , default=2 ,help='下边界权重')
parser.add_argument('--weight_equ1', type=int , default=3 ,help='控制方程1权重')
parser.add_argument('--weight_equ2', type=int , default=2 ,help='控制方程2权重')
parser.add_argument('--weight_equ3', type=int , default=10 ,help='控制方程3权重')
parser.add_argument('--weight_equ4', type=int , default=5 ,help='控制方程4权重')
parser.add_argument('--weight_equ5', type=int , default=10 ,help='控制方程5权重')
parser.add_argument('--weight_equ6', type=int , default=3,help='控制方程6权重')
parser.add_argument('--weight_equ7', type=int , default=3,help='控制方程7权重')
parser.add_argument('--boundary_strictness', type=float , default=0.5 ,help='边界严格参数')
parser.add_argument('--center_value', type=float , default=3 ,help='中心值')
parser.add_argument('--network_MLP', type=str , default="32,32,32,32,32" ,help='全连接网络形状')
#parser.add_argument('--network_MLP', type=str , default="64,64,64,64,64,64,64,64" ,help='全连接网络形状')
parser.add_argument('--check_every', type=int , default=100 ,help='检测周期')
parser.add_argument('--save_dict', type=str , default='run1' ,help='训练文件名')
parser.add_argument('--maxf', type=int , default=10 ,help='端面相对温度最大值')
parser.add_argument('--impose', type=int , default=1 ,help='是否强加Drichlet边界,1为施加')
parser.add_argument('--mtl', type=int , default=0 ,help='是否使用多任务权重学习,1为使用')
parser.add_argument('--log', type=int , default=1 ,help='是否使用对数调整权重')


args = parser.parse_args()
print(args)

save_image_folder = args.save_dict + "-image/"
save_model_folder=args.save_dict + "-model/"

# 确保文件夹存在，如果不存在则创建
if not os.path.exists(save_image_folder):
    os.makedirs(save_image_folder)
if not os.path.exists(save_model_folder):
    os.makedirs(save_model_folder)

use_gpu = args.gpu
device = torch.device("cuda" if use_gpu else "cpu")
if use_gpu:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}

heat_network=torch.load('heat_model.pth',map_location=device)

def load_heat(r,z):
    xx=(r-r1)/(r2-r1)
    yy=z/h1
    x = torch.unsqueeze(xx, dim=1).requires_grad_()
    y = torch.unsqueeze(yy, dim=1).requires_grad_()
    xy = torch.cat((x, y), dim=1)
    DT = heat_network(xy)
    DT = torch.squeeze(DT.requires_grad_()) #*maxf
    DT=(1-yy)*DT+2*xx**3-3*xx**2+1
    return DT *maxf

def calculate_sigma_rr(u,w,r,z):
    #return 2*G * ((1-mu)/(1-2*mu)*diff(u,r) + mu/(1-2*mu)*(u/r+diff(w,z)))
    return 2*G * ((1-mu)/(1-2*mu)*diff(u,r) + mu/(1-2*mu)*(u/r+diff(w,z)))-beta*load_heat(r,z)

def calculate_sigma_theta(u,w,r,z):
    return 2*G * ((1-mu)/(1-2*mu)*u/r + mu/(1-2*mu)*(diff(w,z)+diff(u,r)))-beta*load_heat(r,z)

def calculate_sigma_zz(u,w,r,z):
    return 2*G * ((1-mu)/(1-2*mu)*diff(w,z) + mu/(1-2*mu)*((u/r)+diff(u,r)))-beta*load_heat(r,z)

def calculate_tau_zr(u,w,r,z):
    return G*(diff(w,r)+diff(u,z))  ##MPa  um

def force_balance_r(u,w,r,z):
    sigma_rr=calculate_sigma_rr(u,w,r,z)
    sigma_theta=calculate_sigma_theta(u,w,r,z)
    tau_zr=calculate_tau_zr(u,w,r,z)

    return diff(sigma_rr,r)+diff(tau_zr,z)+(sigma_rr-sigma_theta)/r

def force_balance_z(u,w,r,z):
    sigma_zz=calculate_sigma_zz(u,w,r,z)
    tau_zr=calculate_tau_zr(u,w,r,z)

    return diff(sigma_zz,z)+diff(tau_zr,r)+tau_zr/r

def energy(uu,r,z):
    sigma_rr=calculate_sigma_rr(uu[:,0],uu[:,1],r,z)
    sigma_theta=calculate_sigma_theta(uu[:,0],uu[:,1],r,z)
    tau_zr=calculate_tau_zr(uu[:,0],uu[:,1],r,z)
    sigma_zz=calculate_sigma_zz(uu[:,0],uu[:,1],r,z)

    er=(sigma_rr-mu*(sigma_theta+sigma_zz))/E
    et=(sigma_theta-mu*(sigma_rr+sigma_zz))/E
    ez=(sigma_zz-mu*(sigma_theta+sigma_rr))/E
    ezr=tau_zr/G

    num=int(np.sqrt(len(r)))
    dr=(r2-r1)/num
    dz=h1/num

    U=sum((sigma_rr*er+sigma_zz*ez+sigma_theta*et+tau_zr*ezr)*r*dr*dz)
    W=sum(-1*r1*(-10)*uu[:,0].reshape(num,num)[0]*dz)+\
        sum(r2*(-1)*uu[:,0].reshape(num,num)[-1]*dz)+\
        sum(sigma_zz.reshape(num,num)[:,-1]*uu[:,1].reshape(num,num)[:,-1]*r.reshape(num,num)[:,-1]*dr)
    
    return U-W

p_left=-10 #MPa
p_right=-1 #MPa

#left
boundary_left=BoundaryCondition(
    form=lambda u,x,y: abs(calculate_sigma_rr(u[:,0],u[:,1],x,y))+10*abs(calculate_tau_zr(u[:,0],u[:,1],x,y)), #- (y-h1)*1000,  #-p_left,
    #points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(0.0, 0.0), end=(0.0, 1.0),device=device),
    points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(r1, 0.0), end=(r1, h1),device=device),
    weight=args.weight_left,
    impose=1,
)

#bottom
boundary_bottom=BoundaryCondition(
    form=lambda u,x,y: u[:,1],
    points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(r1, 0.0), end=(r2, 0.0),device=device),
    #points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(0.0, 0.0), end=(1.0, 0.0),device=device),
    weight=args.weight_bottom,
    impose=1,
)

#right
boundary_right=BoundaryCondition(
    #form=lambda u,x,y: calculate_sigma_rr(u[:,0],u[:,1],x,y)-p_right,
    form=lambda u,x,y: abs(calculate_sigma_rr(u[:,0],u[:,1],x,y))+10*abs(calculate_tau_zr(u[:,0],u[:,1],x,y)),
    points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(r2, 0.0), end=(r2, h1),device=device),
    #points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(1.0, 0.0), end=(1.0, 1.0),device=device),
    weight=args.weight_right,
    impose=1,
)

#up
boundary_up=BoundaryCondition(
    form=lambda u,x,y: abs(calculate_sigma_zz(u[:,0],u[:,1],x,y)-((p_left-p_right)*(2*((x-r1)/(r2-r1))**3-3*((x-r1)/(r2-r1))**2+1)+p_right))\
                        +10*abs(calculate_tau_zr(u[:,0],u[:,1],x,y)),
    #form=lambda u,x,y: calculate_sigma_zz(u[:,0],u[:,1],x,y)-10,
    #form=lambda u,x,y: calculate_tau_zr(u[:,0],u[:,1],x,y)-10,
    points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(r1, h1), end=(r2, h1),device=device),
    #points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(0.0, 1.0), end=(1.0, 1.0),device=device),
    weight=args.weight_up,
    impose=1,
)


#观测各边界以及方程的损失
metrics={}
#方程
def equ1(uu,xx,yy):
    error=force_balance_r(uu[:,0],uu[:,1],xx,yy)
    # error=force_balance_r(uu,xx,yy)
    return torch.mean(abs(error)**2)
metrics['equ1']=equ1

def equ2(uu,xx,yy):
    error=force_balance_z(uu[:,0],uu[:,1],xx,yy)
    # error=force_balance_z(uu,xx,yy)
    return torch.mean(abs(error)**2)
metrics['equ2']=equ2

def equ3(uu,xx,yy):
    error=calculate_sigma_rr(uu[:,0],uu[:,1],xx,yy)-uu[:,2]
    return torch.mean(abs(error)**2)
metrics['equ3']=equ3

def equ4(uu,xx,yy):
    error=calculate_sigma_theta(uu[:,0],uu[:,1],xx,yy)-uu[:,3]
    return torch.mean(abs(error)**2)
metrics['equ4']=equ4

def equ5(uu,xx,yy):
    error=calculate_sigma_zz(uu[:,0],uu[:,1],xx,yy)-uu[:,4]
    return torch.mean(abs(error)**2)
metrics['equ5']=equ5

def equ6(uu,xx,yy):
    error=calculate_tau_zr(uu[:,0],uu[:,1],xx,yy)-uu[:,5]
    return torch.mean(abs(error)**2)
metrics['equ6']=equ6

# def equ7(uu,xx,yy):
#     error=energy(uu,xx,yy)
#     return torch.mean(abs(error))
# metrics['equ7']=equ7


fcnn = FCNN(
    n_input_units=2,
    n_output_units=6,
    hidden_units=ast.literal_eval(args.network_MLP),
    actv=nn.Tanh
)

fcnn=fcnn.to(device)

fcnn_approximator = SingleNetworkApproximator2DSpatial(
    single_network=fcnn,
    #single_network=renn,
    pde=(force_balance_r,force_balance_z,calculate_sigma_rr,\
         calculate_sigma_theta,calculate_sigma_zz,calculate_tau_zr),
    boundary_conditions=[
        boundary_left,
        boundary_bottom,
        boundary_right,
        boundary_up
    ],
    boundary_strictness=args.boundary_strictness,
    args=args
)
size_train=args.train_rec_size
adam=optim.Adam(fcnn_approximator.parameters(),lr=args.lr)
#train_gen_spatial = generator_2dspatial_rectangle(size=(size_train, size_train), x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,device=device,random=args.train_gen_random)
train_gen_spatial = generator_2dspatial_rectangle(size=(size_train, size_train), x_min=r1, x_max=r2, y_min=0.0, y_max=h1,device=device,random=args.train_gen_random,bound=True)
valid_gen_spatial = generator_2dspatial_rectangle(size=(50, 50), x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, random=args.valid_gen_random,device=device)

#%matplotlib inline
heat_transfer_2d_solution, _ = _solve_2dspatial(
    train_generator_spatial=train_gen_spatial,
    valid_generator_spatial=valid_gen_spatial,
    approximator=fcnn_approximator,
    optimizer=adam,
    batch_size=args.batch_size,
    max_epochs=args.epochs,
    shuffle=True,
    metrics=metrics,
    monitor=Monitor2DSpatial(        
        # check_on_x=torch.linspace(0.0, 1.0, 50),
        # check_on_y=torch.linspace(0.0, 1.0, 50),
        check_on_x=torch.linspace(r1, r2, 50),
        check_on_y=torch.linspace(0.0, h1, 50),
        check_every=args.check_every,
        device=device,
        args=args
    ),
    device=device,
    args=args
)