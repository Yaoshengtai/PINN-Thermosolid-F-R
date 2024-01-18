import torch 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class Monitor2DSpatial:
    r"""A Monitor for 2D steady-state problems
    """
    def __init__(self, check_on_x, check_on_y, check_every):
        self.using_non_gui_backend = matplotlib.get_backend() == 'agg'

        xy_tensor = torch.cartesian_prod(check_on_x, check_on_y)
        self.xx_tensor = torch.squeeze(xy_tensor[:, 0])
        self.yy_tensor = torch.squeeze(xy_tensor[:, 1])

        self.xx_array = self.xx_tensor.clone().detach().cpu().numpy()
        self.yy_array = self.yy_tensor.clone().detach().cpu().numpy()

        self.check_every = check_every

        self.check_on_x=check_on_x
        self.check_on_y=check_on_y

    def check(self, approximator, history):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        uu_array = approximator(self.xx_tensor, self.yy_tensor).detach().cpu().numpy()
        xx, yy = np.meshgrid(self.check_on_x, self.check_on_y)
        # 创建热力图
        heatmap=axs[0].pcolormesh(xx, yy, uu_array.reshape(xx.shape).T, cmap='rainbow')  # cmap是颜色映射，你可以根据需要选择
        contour_lines = axs[0].contour(xx, yy, uu_array.reshape(xx.shape).T, colors='black', linewidths=0.5,level=15)
        # 添加颜色条
        cbar=plt.colorbar(heatmap,ax=axs[0],label='Temperature')
        # 添加轴标签
        axs[0].set_xlabel('r')
        axs[0].set_ylabel('z')
        axs[0].set_title('Heatmap')
        # 显示图形
        
        axs[1].plot(history['train_loss'], label='training loss')
        axs[1].plot(history['valid_loss'], label='validation loss')
        axs[1].set_title('loss during training')
        axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('loss')
        axs[1].set_yscale('log')

        plt.tight_layout()
        plt.show()
