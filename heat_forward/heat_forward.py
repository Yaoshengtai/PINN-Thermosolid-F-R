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
from PINN.networks import FCNN,Resnet
from PINN.generators import generator_2dspatial_rectangle, generator_2dspatial_segment
from PINN.conditions import BoundaryCondition
from PINN.solvers import SingleNetworkApproximator2DSpatial
from PINN.monitors import Monitor2DSpatial
from PINN.solvers import _solve_2dspatial

r1=0.135 #内径
r2=0.1695  #外径
h1=0.02  #高

h=20602.3271 #对流换热系数
T0=303.15 #外界温度
k=60 #传热系数

maxf=10 #端面最高温度

parser = argparse.ArgumentParser(description='PyTorch Deep Learning Training')

# 添加命令行参数
parser.add_argument('--lr', type=float, default=0.00001, help='学习率')
parser.add_argument('--batch_size', type=int, default=2048, help='批量大小')
parser.add_argument('--epochs', type=int, default=1000000, help='训练轮数')
parser.add_argument('--gpu', type=bool , default=False ,help='使用GPU进行训练')
parser.add_argument('--train_rec_size', type=int , default=128 ,help='矩形区域内生成的点,512*512')
parser.add_argument('--train_bound_size', type=int , default=64 ,help='边界上生成的点数')
parser.add_argument('--train_gen_random', type=bool , default=True ,help='训练生成点是否随机')
parser.add_argument('--valid_gen_random', type=bool , default=True ,help='验证生成点是否随机')
parser.add_argument('--weight_up', type=int , default=0 ,help='上边界权重')
parser.add_argument('--weight_left', type=int , default=2 ,help='左边界权重')
parser.add_argument('--weight_right', type=int , default=5 ,help='右边界权重')
parser.add_argument('--weight_bottom', type=int , default=2 ,help='下边界权重')
parser.add_argument('--boundary_strictness', type=int , default=10 ,help='边界严格参数')
parser.add_argument('--network_MLP', type=str , default="32,32,32,32,32" ,help='全连接网络形状')
parser.add_argument('--check_every', type=int , default=500 ,help='检测周期')
parser.add_argument('--save_dict', type=str , default='run1' ,help='训练文件名')
parser.add_argument('--maxf', type=int , default=10 ,help='端面相对温度最大值')
parser.add_argument('--impose', type=int , default=1 ,help='是否强加Drichlet边界,1为施加')
parser.add_argument('--mtl', type=int , default=1 ,help='是否使用多任务权重学习,1为使用')


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
        
size_train=args.train_rec_size
def heat_transfer(u , xx, yy):
    return diff(u, xx, order=2) + diff(u, yy, order=2)

def heat_transfer_norm(u,xx,yy):
    #return (diff(u,yy,order=2)+diff(u,xx,order=2)/(r2-r1)/(r2-r1)*h1*h1+diff(u,xx)/(xx*(r2-r1)*(r2-r1)+r1*(r2-r1))*h1*h1)
    return diff(u,xx)+maxf*((r2-r1)*xx+r1)/(r2-r1)*diff(u,xx,order=2)+maxf*((r2-r1)*xx+r1)*(r2-r1)/h1/h1*diff(u,yy,order=2)

#left
adiabatic_left=BoundaryCondition(
    form=lambda u, x, y: diff(u,x),
    
    points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(0.0, 0.0), end=(0.0, 1.0),device=device),
    weight=args.weight_left,
    impose=0
)
#bottom
adiabatic_bottom=BoundaryCondition(
    form=lambda u, x, y: diff(u,y),
    #form=lambda u, x, y: u,
    points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(0.0, 0.0), end=(1.0, 0.0),device=device),
    weight=args.weight_bottom,
    impose=0
)
#right
convection_externel=BoundaryCondition(

    form=lambda u, x, y: (r2-r1)*h/k*u+diff(u,x),
    #form=lambda u, x, y: u+k/(r2-r1)*diff(u,x)/h,
    
    points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(1.0, 0.0), end=(1.0, 1.0),device=device),
    #weight=1/h/h/3
    weight=args.weight_right,
    impose=0
)
#up
constant_interface=BoundaryCondition(
    #form=lambda u, x, y: u-torch.sin((1-x)*PI/2)-1,
    #form=lambda u, x, y: u-torch.sin((1-x)*PI/2),
    form=lambda u, x, y: u-(2*x*x*x-3*x*x+1),
    points_generator=generator_2dspatial_segment(size=args.train_bound_size, start=(0.0, 1.0), end=(1.0, 1.0),device=device,random=True),
    weight=args.weight_up,
    impose=args.impose
)

#观测各个边界以及方程的损失
metrics={}
#方程
def pdemse(uu,xx,yy):
    error=heat_transfer_norm(uu,xx,yy)
    return torch.mean(abs(error)**2)
metrics['pdemse']=pdemse

#左边界
def leftbound_mse(uu,xx,yy):
    x,y=next(adiabatic_left.points_generator)
    u=fcnn_approximator.__call__(x.requires_grad_(),y.requires_grad_())
    error=adiabatic_left.form(u,x,y)
    return torch.mean(abs(error)**2)
metrics['leftbound_mse']=leftbound_mse

#下边界
def bottombound_mse(uu,xx,yy):
    x,y=next(adiabatic_bottom.points_generator)
    u=fcnn_approximator.__call__(x.requires_grad_(),y.requires_grad_())
    error=adiabatic_bottom.form(u,x,y)
    return torch.mean(abs(error)**2)
metrics['bottombound_mse']=bottombound_mse

#右边界
def rightbound_mse(uu,xx,yy):
    x,y=next(convection_externel.points_generator)
    u=fcnn_approximator.__call__(x.requires_grad_(),y.requires_grad_())
    error=convection_externel.form(u,x,y)
    return torch.mean(abs(error)**2)
metrics['rightbound_mse']=rightbound_mse
    
# #上边界
# def upbound_mse(uu,xx,yy):
#     x,y=next(constant_interface.points_generator)
#     u=fcnn_approximator.__call__(x.requires_grad_(),y.requires_grad_())
#     error=constant_interface.form(u,x,y)
#     return torch.mean(abs(error)**2)
# metrics['upbound_mse']=upbound_mse

# 与comsol对比mse
def comsol_compare(uu,xx,yy):
    x=torch.linspace(0.0, 1.0, 50).requires_grad_()
    y=torch.linspace(0.0, 1.0, 50).requires_grad_()
    xy_tensor = torch.cartesian_prod(x, y).to(device)
            
    xx = torch.squeeze(xy_tensor[:, 0])
    yy = torch.squeeze(xy_tensor[:, 1])

    uu=fcnn_approximator.__call__(xx,yy)
    uu=uu.detach().cpu().numpy()
    fem=pd.read_csv('./data/h20000.txt',delimiter=r'\s+')
    #fem=pd.read_csv('./data/D_4.txt',delimiter=r'\s+')
    uu_di=abs(uu*args.maxf+303.15-fem['T'].values)
    return np.mean(uu_di **2 )


metrics['comsol_compare']=comsol_compare

    

#pureLoss
# def pureloss(uu,xx,yy):
#     return pdemse(uu,xx,yy)+rightbound_mse(uu,xx,yy)+bottombound_mse(uu,xx,yy)+leftbound_mse(uu,xx,yy)+upbound_mse(uu,xx,yy)
# metrics['pureloss']=pureloss


        

fcnn = FCNN(
    n_input_units=2,
    n_output_units=1,
    hidden_units=ast.literal_eval(args.network_MLP),
    #hidden_units=(32,32),
    #actv=nn.Softplus
    #actv=nn.ReLU
    actv=nn.Tanh
)

fcnn=fcnn.to(device)

renn=Resnet(
    n_input_units=2,
    n_output_units=1,
    hidden_units=(64,64),
    #actv=nn.Tanh
    actv=nn.Softplus
)

fcnn_approximator = SingleNetworkApproximator2DSpatial(
    single_network=fcnn,
    #single_network=renn,
    pde=heat_transfer_norm,
    boundary_conditions=[
        adiabatic_left,
        adiabatic_bottom,
        convection_externel,
        constant_interface
    ],
    boundary_strictness=args.boundary_strictness,
    args=args
)
adam = optim.Adam(fcnn_approximator.parameters(), lr=args.lr)
train_gen_spatial = generator_2dspatial_rectangle(size=(size_train, size_train), x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,device=device,random=args.train_gen_random)
valid_gen_spatial = generator_2dspatial_rectangle(size=(50, 50), x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, random=args.valid_gen_random,device=device)
#%matplotlib inline
heat_transfer_2d_solution, _ = _solve_2dspatial(
    train_generator_spatial=train_gen_spatial,
    valid_generator_spatial=valid_gen_spatial,
    approximator=fcnn_approximator,
    optimizer=adam,
    #optimizer=sgd,
    batch_size=args.batch_size,
    max_epochs=args.epochs,
    shuffle=True,
    metrics=metrics,
    monitor=Monitor2DSpatial(        
        check_on_x=torch.linspace(0.0, 1.0, 50),
        check_on_y=torch.linspace(0.0, 1.0, 50),
        check_every=args.check_every,
        device=device,
        args=args
    ),
    device=device,
    args=args
)