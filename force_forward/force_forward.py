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
beta=alpha *E /(1-2*mu) #热应力系数

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
parser.add_argument('--weight_equ1', type=int , default=5 ,help='控制方程1权重')
parser.add_argument('--weight_equ2', type=int , default=1 ,help='控制方程2权重')
parser.add_argument('--weight_equ3', type=int , default=1 ,help='控制方程3权重')
parser.add_argument('--boundary_strictness', type=float , default=0.5 ,help='边界严格参数')
parser.add_argument('--network_MLP', type=str , default="32,32,32,32,32" ,help='全连接网络形状')
#parser.add_argument('--network_MLP', type=str , default="64,64,64,64,64,64,64,64" ,help='全连接网络形状')
parser.add_argument('--check_every', type=int , default=100 ,help='检测周期')
parser.add_argument('--save_dict', type=str , default='run1' ,help='训练文件名')
parser.add_argument('--maxf', type=int , default=10 ,help='端面相对温度最大值')
parser.add_argument('--impose', type=int , default=1 ,help='是否强加Drichlet边界,1为施加')
parser.add_argument('--mtl', type=int , default=0 ,help='是否使用多任务权重学习,1为使用')



args = parser.parse_args()
print(args)

save_folder = args.save_dict + "-image/"

# 确保文件夹存在，如果不存在则创建
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

use_gpu = args.gpu
device = torch.device("cuda" if use_gpu else "cpu")
if use_gpu:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        
def calculate_sigma_rr(u,w,r,z):
    return 2*G * ((1-mu)/(1-2*mu)*diff(u,r) + mu/(1-2*mu)*(u/r+diff(w,z))) 

def calculate_sigma_theta(u,w,r,z):
    return 2*G * ((1-mu)/(1-2*mu)*u/r + mu/(1-2*mu)*(diff(w,z)+diff(u,r)))

def calculate_sigma_zz(u,w,r,z):
    return 2*G * ((1-mu)/(1-2*mu)*diff(w,z) + mu/(1-2*mu)*((u/r)+diff(u,r)))

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
    return torch.mean(abs(error)**2)
metrics['equ1']=equ1

def equ2(uu,xx,yy):
    error=force_balance_z(uu[:,0],uu[:,1],xx,yy)
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

# def equ3(uu,xx,yy):
#     error=u_accumulate(uu[:,0],uu[:,1],xx,yy)
#     return torch.mean(abs(error)**2)
# metrics['equ3']=equ3

#左边界
# def leftbound_mse(uu,xx,yy):
#     x,y=next(boundary_left.points_generator)
#     u=fcnn_approximator.__call__(x.requires_grad_(),y.requires_grad_())
#     error=boundary_left.form(u,x,y)
#     return torch.mean(abs(error)**2)
# metrics['leftbound_mse']=leftbound_mse

# #下边界
# def bottombound_mse(uu,xx,yy):
#     x,y=next(boundary_bottom.points_generator)
#     u=fcnn_approximator.__call__(x.requires_grad_(),y.requires_grad_())
#     error=boundary_bottom.form(u,x,y)
#     return torch.mean(abs(error)**2)
# metrics['bottombound_mse']=bottombound_mse

#右边界
# def rightbound_mse(uu,xx,yy):
#     x,y=next(boundary_right.points_generator)
#     u=fcnn_approximator.__call__(x.requires_grad_(),y.requires_grad_())
#     error=boundary_right.form(u,x,y)
#     return torch.mean(abs(error)**2)
# metrics['rightbound_mse']=rightbound_mse

#上边界
# def upbound_mse(uu,xx,yy):
#     x,y=next(boundary_up.points_generator)
#     u=fcnn_approximator.__call__(x.requires_grad_(),y.requires_grad_())
#     error=boundary_up.form(u,x,y)
#     return torch.mean(abs(error)**2)
# metrics['upbound_mse']=upbound_mse

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
    pde=(force_balance_r,force_balance_z,calculate_sigma_rr,calculate_sigma_theta,calculate_sigma_zz,calculate_tau_zr),#,calculate_sigma_rr,calculate_sigma_theta,calculate_sigma_zz,calculate_tau_zr
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
train_gen_spatial = generator_2dspatial_rectangle(size=(size_train, size_train), x_min=r1, x_max=r2, y_min=0.0, y_max=h1,device=device,random=args.train_gen_random)
valid_gen_spatial = generator_2dspatial_rectangle(size=(50, 50), x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, random=args.valid_gen_random,device=device)#%matplotlib inline
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