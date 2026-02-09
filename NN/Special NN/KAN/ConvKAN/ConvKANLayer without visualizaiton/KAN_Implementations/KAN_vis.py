import matplotlib.pyplot as plt
import numpy as np
import torch
# from numpy.array_api import arange


def print_parameter_details(model):
    '''
    查看模型层数和参数
    Args:
        model:
    Returns:
    '''
    print(model)
    total_params = 0
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()} {'requires_grad' if param.requires_grad else 'frozen'}")
        total_params += param.numel()
    print(f"Total trainable parameters: {total_params}")


def act_vis(model,l = 0,i = 0,j = 0,symbolic_show=False):
    '''
    提取KAN中某一个激活函数进行可视化
    Args:
        model:kan layer
        l:层索引
        i:输入神经元索引
        j:输出神经元索引
        symbolic_show:是否展示kan的所有符号化图像
    Returns:
    '''
    if symbolic_show:
        model.plot()
        plt.show()
    print(model.spline_preacts[l].shape,model.spline_postacts[l].shape,model.spline_preacts[l][:, j, i].shape)
    # 定义要提取的激活函数所在的层索引 l、输入神经元索引 i 和输出神经元索引 j。从模型中提取激活函数的输入值和输出值。
    # model.spline_preacts[l].shape=model.spline_postacts[l]=torch.Size([1024, 4, 27])
    # 其中1024是依据卷积核的kernel_size从图片上滑动裁剪后得到的区块数量，
    # 4是该层的输出特征数量，27是该层的输入特征数量；
    # torch.Size([1024])
    inputs = model.spline_preacts[l][:, j, i]  # 提取第 l 层，输出神经元索引为 j，输入神经元索引为 i 的激活函数的输入值。
    # torch.Size([1024])
    outputs = model.spline_postacts[l][:, j, i]  # 提取第 l 层，输出神经元索引为 j，输入神经元索引为 i 的激活函数的输出值。
    print("range",model.get_range(l, i, j,verbose=False))  # 获取激活函数的范围
    # 由于提取的输入值可能不是有序的，因此需要对输入值进行排序，并根据排序索引重新排列输入值和输出值。
    rank = np.argsort(inputs)  # 获取输入值的排序索引。
    inputs = inputs[rank]  # 根据排序索引重新排列输入值和输出值，确保输入值是有序的。
    outputs = outputs[rank]
    plt.plot(inputs, outputs, marker="o")  # 使用圆形标记每个数据点，以便更清晰地展示输入值和输出值之间的关系。
    plt.show()


def conv_kernal_act(model,l=0,in_channel=0,out_channel=0,kernal_size=3):
    '''
    提取某个卷积核里的所有激活函数进行绘图
    Args:
        model:kan layer
        l:层索引。默认0，也只能是0，因为此处KAN只有1层
        in_channel：提取哪个输入通道的激活函数，也就是i，即输入神经元索引
        out_channel：提取哪个输出通道的激活函数，也就是j，即输出神经元索引
    Returns:
    '''
    # Prepare the figure for subplots
    fig, axs = plt.subplots(kernal_size, kernal_size,constrained_layout=True, figsize=(12, 15)) #  Width, height in inches.
    '''
    model.spline_preacts[l]：提取该卷积层中所有的激活函数数量，torch.Size([1024, 4, 27])
    其中1024是依据卷积核的kernel_size从图片上滑动裁剪后得到的区块数量，
    4是该层的输出特征数量，27是该层的输入特征数量；
    '''
    # Loop through the first 9 kernels
    for idx in range(kernal_size**2):
        ax = axs[idx // kernal_size, idx % kernal_size]  # Get the appropriate subplot
        # 提取第 l 层，输出神经元索引为 j/out_channel，输入神经元索引为 in_channel*kernal_size**2+idx 的激活函数的输入值。
        # 之所以in_channel*kernal_size**2+idx，是因为在N通道输入图像中，其N个卷积核都是按顺序排列的
        # 因此第i个卷积核的所有激活函数就需要按顺序进行提取
        inputs = model.spline_preacts[l][:, out_channel, in_channel*kernal_size**2+idx]  # torch.Size([1024])
        # 提取第 l 层，输出神经元索引为 j/out_channel，输入神经元索引为 i 的激活函数的输出值。
        outputs = model.spline_postacts[l][:, out_channel, in_channel*kernal_size**2+idx]  # torch.Size([1024])
        # 由于提取的输入值可能不是有序的，因此需要对输入值进行排序，并根据排序索引重新排列输入值和输出值。
        rank = np.argsort(inputs)  # 获取输入值的排序索引。
        inputs = inputs[rank]  # 根据排序索引重新排列输入值和输出值，确保输入值是有序的。
        outputs = outputs[rank]
        ax.plot(inputs, outputs)  # Plot the spline of the kernel
        ax.set_title(f'Element {idx + 1}')  # Set the title for each subplot
        ax.set_xlabel("Input Mangitude")  # Add x-label
        ax.set_ylabel("Output Mangitude")  # Add y-label
        ax.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    # plt.tight_layout()  # Adjust layout
    plt.show()  # Display the plots


# Note this is just a rough demo for Visualization. Need modifcation.
def visualize_kan_parameters(kan_layer, l=0,layer_name="layer_name",type="origin"):
    '''
    对激活函数系数进行可视化
    Args:
        # data:输入kanlayer的数据[batch_size*num_patches, in_channels*kernel_size*kernel_size]
        kan_layer:
        layer_name:
        type:哪种KAN，不同KAN操作不同
        l:层索引
        i:输入神经元索引
        j:输出神经元索引
    Returns:
    '''
    if type=="efficient":
        base_weights = kan_layer.base_weight.data.cpu().numpy()
        # ravel：Return a flattened array.
        plt.hist(base_weights.ravel(), bins=50)
        plt.title(f"Distribution of Base Weights - {layer_name}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.show()
        if hasattr(kan_layer, 'spline_weight'):
            spline_weights = kan_layer.spline_weight.data.cpu().numpy()
            plt.hist(spline_weights.ravel(), bins=50)
            plt.title(f"Distribution of Spline Weights - {layer_name}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plt.show()
    elif type=="origin":
        # 老版KAN的shape：(number of splines, number of coef params)
        spline_weights = kan_layer.act_fun[l].coef.data.cpu()
        plt.hist(spline_weights.ravel(), bins=50)
        plt.title(f"Distribution of Spline Weights - {layer_name}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.show()

        # 老版KAN的shape：(number of splines, number of grid points)
        # grid=kan_layer.act_fun[l].grid.data.cpu()
        # print(data.shape,spline_weights.shape,grid.shape)
        # (batch, in_dim, out_dim).里面都是训练的激活函数的系数
        # out=coef2curve(data,grid,spline_weights,k)
        # plt.plot(out[0,i,j])
        # plt.show()

    else:
        raise ValueError(f"valid {type}")




def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases

    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde

    Returns:
    --------
        spline values : 3D torch.tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> B_batch(x, grids, k=k).shape
    torch.Size([5, 13, 100])
    '''

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value


def coef2curve(x_eval, grid, coef, k, device="cpu"):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 2D torch.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
    >>> coef2curve(x_eval, grids, coef, k=k).shape
    torch.Size([5, 100])
    '''
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    y_eval = torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k, device=device))
    return y_eval


