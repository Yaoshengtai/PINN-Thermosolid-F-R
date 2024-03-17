import torch
from abc import ABC, abstractmethod
from torch.autograd import grad
from torch import optim
import math
import json
from PINN.function import *



class Approximator(ABC):
    r"""The base class of approximators. An approximator is an approximation of the differential equation's solution.
    It knows the parameters in the neural network, and how to calculate the loss function and the metrics.
    """
    @abstractmethod
    def __call__(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def parameters(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def calculate_loss(self):
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def calculate_metrics(self):
        raise NotImplementedError  # pragma: no cover

class SingleNetworkApproximator2DSpatial(Approximator):
    r"""An approximator to approximate the solution of a 2D steady-state problem.
    The boundary condition will be enforced by a regularization term in the loss function.

    :param single_network: A neural network with 2 input nodes (x, y) and 1 output node.
    :type single_network: `torch.nn.Module`
    :param pde: The PDE to solve. If the PDE is :math:`F(u, x, y) = 0` then `pde`
        should be a function that maps :math:`(u, x, y)` to :math:`F(u, x, y)`.
    :type pde: function
    :param boundary_conditions: A list of boundary conditions.
    :type boundary_conditions: list[`temporal.BoundaryCondition`]
    :param boundary_strictness: The regularization parameter, defaults to 1.
        A larger regularization parameter enforces the boundary conditions more strictly.
    :type boundary_strictness: float
    """

    def __init__(self, single_network, pde, boundary_conditions, boundary_strictness,args):
        self.single_network = single_network
        self.pde = pde
        self.boundary_conditions = boundary_conditions
        self.boundary_strictness = boundary_strictness
        self.args=args
        

    def __call__(self, xx, yy):
        x = torch.unsqueeze(xx, dim=1).requires_grad_()
        y = torch.unsqueeze(yy, dim=1).requires_grad_()
        xy = torch.cat((x, y), dim=1)
        uu = self.single_network(xy)
        uu = torch.squeeze(uu.requires_grad_())  # Ensure that uu also requires gradients
        ## exact imposition diriclet boundary
        
        if self.args.impose!=0:
            u_par_0=0
            #uu=u_par+(1-yy)*yy*(1-xx)*xx*uu
            uu[:,1]=u_par_0+yy*uu[:,1].clone()
            uu[:,5]=u_par_0+(h1-yy)*(xx-r1)*(r2-xx)*yy*uu[:,5].clone()

            u_par_up=((-10+1)*(2*((xx-r1)/(r2-r1))**3-3*((xx-r1)/(r2-r1))**2+1)-1)
            uu[:,4]=u_par_up+(h1-yy)*uu[:,4].clone()
            uu[:,2]=u_par_up+(xx-r1)*(r2-xx)*uu[:,2].clone()

        return uu
    
    def parameters(self):
        return self.single_network.parameters()

    def calculate_loss(self, xx, yy):
        uu = self.__call__(xx, yy)
        #print(uu.shape)
        # print(uu[:,0].shape)
        equation_mse = self.args.weight_equ1*torch.mean(abs(self.pde[0](uu[:,0],uu[:,1], xx, yy))**2)+\
                       self.args.weight_equ2*torch.mean(abs(self.pde[1](uu[:,0],uu[:,1], xx, yy))**2)+\
                       self.args.weight_equ3*torch.mean(abs(self.pde[2](uu[:,0],uu[:,1], xx, yy)-uu[:,2])**2)+\
                       self.args.weight_equ4*torch.mean(abs(self.pde[3](uu[:,0],uu[:,1], xx, yy)-uu[:,3])**2)+\
                       self.args.weight_equ5*torch.mean(abs(self.pde[4](uu[:,0],uu[:,1], xx, yy)-uu[:,4])**2)+\
                       self.args.weight_equ6*torch.mean(abs(self.pde[5](uu[:,0],uu[:,1], xx, yy)-uu[:,5])**2)#+\
                       #self.args.weight_equ7*torch.mean(abs(self.pde[6](uu, xx, yy)))   #rr,theta,zz,zr
                
        #boundary_mse = self.boundary_strictness * sum(self._boundary_mse(bc) for bc in self.boundary_conditions)
        #h1=0.02  #高
        #weight_pde=h1 ** 4 /3
        #weight_pde=0.2
        return equation_mse #+ boundary_mse

    def _boundary_mse(self, bc):
        xx, yy = next(bc.points_generator)
        uu= self.__call__(xx.requires_grad_(), yy.requires_grad_())
        #print(uu.shape)
        loss=torch.mean(abs(bc.form(uu, xx, yy))**2)
        w=bc.weight
        loss=loss*w
        return loss
    
    def update_weight(self,pde_mean_grad_equ1,equ_list,weight,device):
        #weight=torch.tensor([1 for i in range(len(equ_list))]).to(device)
        while True:
            alpha=0.1
            weight_hat=abs(torch.log2(pde_mean_grad_equ1/equ_list))
            weight=(1-alpha)*weight+alpha*weight_hat
            yield weight

    def calculate_weight(self,xx,yy,weight,device):
        uu = self.__call__(xx, yy)
        equ_list=[]
        equ1_mse = torch.mean(abs(self.pde[0](uu[:,0],uu[:,1], xx, yy))**2)
        pde_grad=grad(equ1_mse,self.single_network.parameters(),create_graph=False,allow_unused=True,retain_graph=True)
        pde_grad = torch.cat([grad_tensor.view(-1) for grad_tensor in pde_grad if grad_tensor is not None])    
        pde_mean_grad_equ1=(torch.mean(abs(pde_grad)))

    
        equ2_mse = torch.mean(abs(self.pde[1](uu[:,0],uu[:,1], xx, yy))**2)
        pde_grad=grad(equ2_mse,self.single_network.parameters(),create_graph=False,allow_unused=True,retain_graph=True)
        pde_grad = torch.cat([grad_tensor.view(-1) for grad_tensor in pde_grad if grad_tensor is not None])    
        equ_list.append(torch.mean(abs(pde_grad)))

        for i in range(2,len(self.pde)):
            equation_mse = torch.mean(abs(self.pde[i](uu[:,0],uu[:,1], xx, yy)-uu[:,i])**2)
            pde_grad=grad(equation_mse,self.single_network.parameters(),create_graph=False,allow_unused=True,retain_graph=True)
            pde_grad = torch.cat([grad_tensor.view(-1) for grad_tensor in pde_grad if grad_tensor is not None])    
            equ_list.append(torch.mean(abs(pde_grad)))

        equ_list=torch.tensor(equ_list).to(device)
        #print(pde_mean_grad.item())

        # for bc in self.boundary_conditions:
        #     if bc.impose==0:
        #         xx,yy=next(bc.points_generator)
        #         uu= self.__call__(xx.requires_grad_(), yy.requires_grad_())
        #         loss=torch.mean(abs(bc.form(uu, xx, yy)))**2
        #         bc_grad=grad(loss,self.single_network.parameters(),create_graph=False,allow_unused=True)
        #         bc_grad = torch.cat([grad_tensor.view(-1) for grad_tensor in bc_grad if grad_tensor is not None])    
        #         bc_mean_grad=torch.mean(abs(bc_grad))
                
        #         equ_list.append(torch.tensor(bc_mean_grad).to(device))
        #print(boundary_mean_grad)
        #weight=torch.tensor([1 for i in range(len(equ_list))]).to(device)
        #weight=next(self.update_weight(pde_mean_grad_equ1,equ_list,weight,device))
        weight=next(self.update_weight(pde_mean_grad_equ1,equ_list,weight,device))
        #print(weight)
        return weight
    
    def calculate_loss_mtl(self,xx,yy,weight,device):

        uu = self.__call__(xx, yy)
        
        equ1_mse = torch.mean(abs(self.pde[0](uu[:,0],uu[:,1], xx, yy))**2)
        beta=self.args.weight_equ1
        Loss=equ1_mse*beta

        weight=self.calculate_weight(xx,yy,weight,device)

        loss=torch.mean(abs(self.pde[1](uu[:,0],uu[:,1], xx, yy))**2)  #*beta
        Loss+=loss*weight[0]

        for i in range(2,len(self.pde)):
            loss=torch.mean(abs(self.pde[i](uu[:,0],uu[:,1], xx, yy)-uu[:,i])**2)
            Loss+=loss*weight[i-1]
        
        # for bc in self.boundary_conditions:
        #     if bc.impose==0:
        #         i=len(self.pde)-1
        #         xx,yy=next(bc.points_generator)
        #         uu= self.__call__(xx.requires_grad_(), yy.requires_grad_())
        #         loss=torch.mean(abs(bc.form(uu, xx, yy))**2)
        #         Loss+=loss*weight[i]
        #         i+=1
        return weight,Loss


    def calculate_metrics(self, xx, yy, metrics):
        uu = self.__call__(xx, yy)
        return {
            metric_name: metric_func(uu,xx,yy)
            for metric_name, metric_func in metrics.items()
        }

def _train_2dspatial(train_generator_spatial, train_generator_temporal,
                     approximator, optimizer, metrics, shuffle, batch_size,device,args):
    xx, yy = next(train_generator_spatial)
    xx,yy=xx.to(device),yy.to(device)

    xx.requires_grad = True
    yy.requires_grad = True
    training_set_size = len(xx)
    idx = torch.randperm(training_set_size) if shuffle else torch.arange(training_set_size)
    weight=torch.tensor([1 for i in range(len(approximator.pde)-1)]).to(device)
    batch_start, batch_end = 0, batch_size
    while batch_start < training_set_size:
        if batch_end > training_set_size:
            batch_end = training_set_size
        batch_idx = idx[batch_start:batch_end]
        batch_xx = xx[batch_idx].to(device)
        batch_yy = yy[batch_idx].to(device)
        if args.mtl==0:
            batch_loss = approximator.calculate_loss(batch_xx, batch_yy)
        else:
            weight,batch_loss=approximator.calculate_loss_mtl(batch_xx, batch_yy,weight,device)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        batch_start += batch_size
        batch_end += batch_size
    # if args.mtl==0:
    #     epoch_loss = approximator.calculate_loss(xx,yy)
    # else:
    #     weight,epoch_loss = approximator.calculate_loss_mtl(xx, yy,weight,device)
    epoch_loss=batch_loss
    
    epoch_metrics = approximator.calculate_metrics(xx, yy, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics,weight


# validation phase for 2D steady-state problems
def _valid_2dspatial(valid_generator_spatial, valid_generator_temporal, approximator, metrics):
    xx, yy = next(valid_generator_spatial)
    xx.requires_grad = True
    yy.requires_grad = True

    epoch_loss = approximator.calculate_loss(xx, yy).item()

    epoch_metrics = approximator.calculate_metrics(xx, yy, metrics)
    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item()

    return epoch_loss, epoch_metrics

def _solve_2dspatial(
    train_generator_spatial, valid_generator_spatial,
    approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,device,args,
):
    r"""Solve a 2D steady-state problem

    :param train_generator_spatial:
        A generator to generate 2D spatial points for training.
    :type train_generator_spatial: generator
    :param valid_generator_spatial:
        A generator to generate 2D spatial points for validation.
    :type valid_generator_spatial: generator
    :param approximator:
        An approximator for 2D time-state problem.
    :type approximator: `temporal.SingleNetworkApproximator2DSpatial`, `temporal.SingleNetworkApproximator2DSpatialSystem`, or a custom `temporal.Approximator`
    :param optimizer:
        The optimization method to use for training.
    :type optimizer: `torch.optim.Optimizer`
    :param batch_size:
        The size of the mini-batch to use.
    :type batch_size: int
    :param max_epochs:
        The maximum number of epochs to train.
    :type max_epochs: int
    :param shuffle:
        Whether to shuffle the training examples every epoch.
    :type shuffle: bool
    :param metrics:
        Metrics to keep track of during training.
        The metrics should be passed as a dictionary where the keys are the names of the metrics,
        and the values are the corresponding function.
        The input functions should be the same as `pde` (of the approximator) and the output should be a numeric value.
        The metrics are evaluated on both the training set and validation set.
    :type metrics: dict[string, callable]
    :param monitor:
        The monitor to check the status of nerual network during training.
    :type monitor: `temporal.Monitor2DSpatial` or `temporal.MonitorMinimal`
    """
    return _solve_spatial_temporal(
        train_generator_spatial, None, valid_generator_spatial, None,
        approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,device,args,
        train_routine=_train_2dspatial, valid_routine=_valid_2dspatial
    )


# _solve_1dspatial_temporal, _solve_2dspatial_temporal, _solve_2dspatial all call this function in the end
def _solve_spatial_temporal(
    train_generator_spatial, train_generator_temporal, valid_generator_spatial, valid_generator_temporal,
    approximator, optimizer, batch_size, max_epochs, shuffle, metrics, monitor,device,args,
    train_routine, valid_routine
):
    history = {'train_loss': [], 'valid_loss': []}
    for metric_name, _ in metrics.items():
        history['train_' + metric_name] = []
        #history['valid_' + metric_name] = []
    with open(args.save_dict+'-train_log.txt', 'w') as file:
        file.write("....... begin training ....... \n")
    for epoch in range(max_epochs):
        train_epoch_loss, train_epoch_metrics,weight = train_routine(
            train_generator_spatial, train_generator_temporal, approximator, optimizer, metrics, shuffle, batch_size,device,args,
        )
        history['train_loss'].append(train_epoch_loss.detach().cpu())
        
        for metric_name, metric_value in train_epoch_metrics.items():
            history['train_' + metric_name].append(metric_value)

        # valid_epoch_loss, valid_epoch_metrics = valid_routine(
        #     valid_generator_spatial, valid_generator_temporal, approximator, metrics
        # )
        #history['valid_loss'].append(valid_epoch_loss)
        # for metric_name, metric_value in valid_epoch_metrics.items():
        #     history['valid_' + metric_name].append(metric_value)
        # if epoch >4000:
        #     optimizer=optim.LBFGS(approximator.parameters(),lr=0.01)
        # if epoch >500:
        #     args.mtl=1
        # else:
        #     args.mtl=0
        if monitor and epoch % monitor.check_every == 0:
            monitor.check(approximator, history,epoch)

        #print("\r"+"Already calculate for "+ str(epoch) + "/"+str(max_epochs),end='')
        if epoch %1==0:
            with open(args.save_dict+'-train_log.txt', 'w') as file:
                last_items = {key: values[-1] if values else None for key, values in history.items()}
                for key, value in last_items.items():
                    file.write(f"{key}: {value}\n")
                file.write("weight: left bottom right \n")
                for w in weight.detach().cpu().tolist():
                    file.write(str(w)+'\n')
                file.write("Already calculate for "+ str(epoch) + "/"+str(max_epochs)+'\n')
        

        #print("Already calculate for "+ str(epoch) + "/"+str(max_epochs))
        # if epoch % 1000==0:
        #     print("Already calculate for "+ str(epoch) + "/"+str(max_epochs))

    return approximator, history