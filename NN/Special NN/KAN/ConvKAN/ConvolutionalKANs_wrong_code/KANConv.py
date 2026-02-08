import torch
import math
import sys
sys.path.append('./kan_convolutional')
from KANLinear import KANLinear
import convolution


#Script que contiene la implementación del kernel con funciones de activación.
# 包含带有激活函数的内核实现的脚本。
class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(self,in_channels: int = 1,out_channels: int = 1,
            kernel_size: tuple = (2,2),stride: tuple = (1,1),padding: tuple = (0,0),
            dilation: tuple = (1,1),grid_size: int = 5,spline_order:int = 3,
            scale_noise:float = 0.1,scale_base: float = 1.0,scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,grid_eps: float = 0.02,grid_range: tuple = [-1, 1],
            device: str = "cpu"):
        """
        Kan Convolutional Layer with multiple convolutions
        建立in_channels*out_channels个卷积核特征图,即单通道卷积核
        Args:
            n_convs (int): Number of convolutions to apply
            kernel_size (tuple): Size of the kernel
            stride (tuple): Stride of the convolution
            padding (tuple): Padding of the convolution
            dilation (tuple): Dilation of the convolution
            grid_size (int): Size of the grid
            spline_order (int): Order of the spline
            scale_noise (float): Scale of the noise
            scale_base (float): Scale of the base - 基础尺度
            scale_spline (float): Scale of the spline - 样条的尺度
            base_activation (torch.nn.Module): Activation function - 基础激活函数
            grid_eps (float): Epsilon of the grid - 网格的 epsilon 值
            grid_range (tuple): Range of the grid - 网格的范围
            device (str): Device to use
        """
        super(KAN_Convolutional_Layer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.spline_order = spline_order # 卷积核的里每个可学习激活函数的阶数
        self.kernel_size = kernel_size
        # self.device = device
        self.dilation = dilation
        self.padding = padding
        self.convs = torch.nn.ModuleList() # # 卷积层列表，用于存储每个单通道的卷积核，即卷积核的特征图
        self.stride = stride

        # Create n_convs KAN_Convolution objects
        # 创建 in_channels*out_channels 个 KAN_Convolution 对象，并将它们添加到卷积层列表中
        # 此处可以考虑改写为解耦KAN。即depth-wise
        for _ in range(in_channels*out_channels):
            self.convs.append(KAN_Convolution(kernel_size= kernel_size,stride = stride,padding=padding,dilation = dilation,
                    grid_size=grid_size,spline_order=spline_order,scale_noise=scale_noise,
                    scale_base=scale_base,scale_spline=scale_spline,base_activation=base_activation,
                    grid_eps=grid_eps,grid_range=grid_range, # device = device ## changed device to be allocated as per the input device for pytorch DDP
                ))

    def forward(self, x: torch.Tensor):
        # If there are multiple convolutions, apply them all
        self.device = x.device
        #if self.n_convs>1:
        return convolution.multiple_convs_kan_conv2d(x, self.convs,self.kernel_size[0],self.out_channels,self.stride,self.dilation,self.padding,self.device)
        
        # If there is only one convolution, apply it
        #return self.convs[0].forward(x)
        

class KAN_Convolution(torch.nn.Module):
    def __init__(
            self,
            kernel_size: tuple = (2,2),
            stride: tuple = (1,1),
            padding: tuple = (0,0),
            dilation: tuple = (1,1),
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=torch.nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: tuple = [-1, 1],
            device = "cpu"
        ):
        """
        Args
        """
        super(KAN_Convolution, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.device = device
        self.conv = KANLinear(
            in_features = math.prod(kernel_size),
            out_features = 1,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range
        )

    def forward(self, x: torch.Tensor):
        self.device = x.device
        return convolution.multiple_convs_kan_conv2d(x, [self],self.kernel_size[0],1,self.stride,self.dilation,self.padding,self.device)
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum( layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)



