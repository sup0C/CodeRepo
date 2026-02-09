import torch
from ConvKAN import ConvKAN
import matplotlib.pyplot as plt
import numpy as np
from KAN_Implementations.KAN_vis import (visualize_kan_parameters,
                                         print_parameter_details,
                                         act_vis)

import sys
sys.path.append("./")



# Note this is just a rough demo for Visualization. Need modifcation.
# def visualize_kan_parameters(kan_layer, layer_name):
#     base_weights = kan_layer.base_weight.data.cpu().numpy()
#     plt.hist(base_weights.ravel(), bins=50)
#     plt.title(f"Distribution of Base Weights - {layer_name}")
#     plt.xlabel("Weight Value")
#     plt.ylabel("Frequency")
#     plt.show()
#     if hasattr(kan_layer, 'spline_weight'):
#         spline_weights = kan_layer.spline_weight.data.cpu().numpy()
#         plt.hist(spline_weights.ravel(), bins=50)
#         plt.title(f"Distribution of Spline Weights - {layer_name}")
#         plt.xlabel("Weight Value")
#         plt.ylabel("Frequency")
#         plt.show()

# def print_parameter_details(model):
#     print(model)
#     total_params = 0
#     for name, param in model.named_parameters():
#         print(f"{name}: {param.size()} {'requires_grad' if param.requires_grad else 'frozen'}")
#         total_params += param.numel()
#     print(f"Total trainable parameters: {total_params}")

# def act_vis(model,l = 0,i = 2,j = 0):
#     '''
#     提取KAN中某一个激活函数进行可视化
#     Args:
#         model:
#         l:层索引
#         i:输入神经元索引
#         j:输出神经元索引
#     Returns:
#     '''
#     model.plot()
#     plt.show()
#
#     # 定义要提取的激活函数所在的层索引 l、输入神经元索引 i 和输出神经元索引 j。从模型中提取激活函数的输入值和输出值。
#     inputs = model.spline_preacts[l][:, j, i]  # 提取第 l 层，输出神经元索引为 j，输入神经元索引为 i 的激活函数的输入值。
#     outputs = model.spline_postacts[l][:, j, i]  # 提取第 l 层，输出神经元索引为 j，输入神经元索引为 i 的激活函数的输出值。
#     print("range",model.get_range(l, i, j,verbose=False))  # 获取激活函数的范围
#     # 由于提取的输入值可能不是有序的，因此需要对输入值进行排序，并根据排序索引重新排列输入值和输出值。
#     rank = np.argsort(inputs)  # 获取输入值的排序索引。
#     inputs = inputs[rank]  # 根据排序索引重新排列输入值和输出值，确保输入值是有序的。
#     outputs = outputs[rank]
#     plt.plot(inputs, outputs, marker="o")  # 使用圆形标记每个数据点，以便更清晰地展示输入值和输出值之间的关系。
#     plt.show()

if __name__ == '__main__':
    model = ConvKAN(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1,
                   version="Original", grid_size=3,seed=2) # 共需要训练in_channels*kernel_size*kernel_size乘out_channels个激活函数

    x = torch.rand((1,3,32,32))
    y = model(x)
    print("*"*100)
    print("Input: ", x.shape)
    print("Output: ", y.shape,y[0,0,0,:5])
    # print_parameter_details(model)
    KAN=model.mykan # model.mykan也就是KAN，从model中提取出KAN层
    visualize_kan_parameters(KAN)
    act_vis(KAN)

    # 保存并重加载模型
    # torch.save(model.state_dict(), 'model.pth')
    # model = ConvKAN(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1,
    #                version="Original", grid_size=3,seed=1)
    # model.load_state_dict(torch.load('model.pth',weights_only=True,map_location=torch.device("cpu")), strict=False)
    # y = model(x) # 必须走一遍数据才能刷新出来KAN layer的属性，否则网络会直接去nn.module中找KAN的类属性
    # KAN=model.mykan # model.mykan也就是KAN，从model中提取出KAN层
    # act_vis(KAN)

