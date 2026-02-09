import matplotlib.pyplot as plt
from kan import *
import torch
import numpy as np
from streamlit import title

# torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(device)

# 创建数据
# x = np.linspace(-1, 1, 500).reshape(-1, 1)
# y = np.sin(np.pi * x) + np.random.normal(0, 0.1, x.shape)
# # 转换数据为PyTorch张量
# x_tensor = torch.FloatTensor(x).to(device)
# y_tensor = torch.FloatTensor(y).to(device)
# dataset = {}
# dataset['train_input'] = x_tensor
# dataset['test_input'] = x_tensor
# dataset['train_label'] = y_tensor
# dataset['test_label'] = y_tensor
# print("dataset['train_input'].shape",dataset['train_input'].shape) # torch.Size([500, 1])
# print(dataset['train_label'].shape) # torch.Size([500, 1])

# 创建数据2
from kan.utils import create_dataset
# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)
print(dataset['train_input'].shape,dataset['test_input'].shape, dataset['train_label'].shape,dataset['test_label'].shape)

# 1 创建一个 KAN：第一层是1个特征输入层节点（x），即输入特征数为2；
# 5表示5 neurons in the first hidden layer；1表示1 output node；
# k = 3 代表选择三次样条曲线，grid = 3 代表网格点为 3
# β 控制激活函数的透明度。较大的 β 意味着更多的激活函数会显现出来。我们通常希望设置一个合适的 β，
# 使得只有重要的连接在视觉上是显著的。透明度被设定为 tanh(β∣ϕ∣1)，其中∣ϕ∣1是激活函数的L1范数。默认β=3。
beta=100 # beta通常用来控制绘图时的细节或正则化参数
model = KAN(width=[2,5,1], grid=3, k=3, seed=42,
        # 初始化所有激活函数为完全线性，或者 base_fun = "identity",需要搭配noise_scale = 0，默认：'silu'
        # an activation function phi(x) = sb_scale * b(x) + sp_scale * spline(x)
        # scale_base:mangitude of the base function b(x)
        # scale_base_mu:magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_mu
        # scale_base_sigma: float. magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_sigma
        # scale_sp: mangitude of the spline function spline(x)
        base_fun = lambda x: x, # base_fun是residual function or the base function b(x)
        # initial injected noise to spline.
        noise_scale = 0, # 当noise_scale设置为较大的数值，则是使用噪声对样条进行初始化
        #  sparse initialization (True) or normal dense initialization. Default: False.
        sparse_init=True, # 稀疏初始化,若为True，则大多数 scale_base 和 scale_sp 将设置为零
        auto_save=True,device=device) # width的len代表深度，从左到右依次为每层的神经元个数。NOTE：第一层神经元个数应该和特征数相同
model(dataset['train_input'])
# plot的量度参数metric：metric='forward_n'、'forward_u'、'backward'
model.plot(beta=beta,scale=0.5, # “scale” 参数调整图窗的大小。默认情况下：0.5
            title = 'My KAN',
           sample=True, # 即使用使用训练数据的离散点绘制激活函数，即显示样本分布
           in_vars=[r'$\alpha$', 'x'], # the name(s) of input variables
           out_vars=['y'],) # Plot KAN at initialization.
plt.show()

# 2 节点（神经元）的索引 - 移除某个神经元
model.remove_node(1,2) # 从模型中移除第 2 层的第 3 个神经元。删除后，该神经元的所有入边和出边对应的激活函数都会被禁用或设为零，从而在模型中“剪枝”掉这个节点。
model.plot(beta=beta)
plt.show()

# 3  层索引
# KAN模型中每一层都有两个对应的部分，一个是数值计算用的 spline 层，另一个是符号计算或解释用的符号层。
# 这段代码依次打印前三层中 spline 层和符号层的输入维度（in_dim）和输出维度（out_dim），用于确认各层的结构设置是否正确。
for i in range(2):
    print(f"{i}"*100) # 此处KAN为[2,5,1]
    # 激活层
    # act_fun:a list of KANLayers，存储了每一层基于 spline（B‐Spline 激活函数）的激活层
    print("model.act_fun[i]", model.act_fun[i])  # => KAN Layer (Spline)
    # symbolic_fun： a list of Symbolic_KANLayer。存储了对应的符号层，用于符号化处理或解释性计算
    print("model.symbolic_fun[i]", model.symbolic_fun[i])  # => KAN Layer (Symbolic)
    print("act_fun in_dim",model.act_fun[i].in_dim, model.act_fun[i].out_dim) # 2 5
    print("symbolic_fun in_dim",model.symbolic_fun[i].in_dim, model.symbolic_fun[i].out_dim) # 2 5
    # 获取第 i 层 spline 层的网格信息，定义了 B-spline 的节点分布，用于决定激活函数的分段结构。
    # size=(in_dim, G+2k+1). G: the number of grid intervals; k: spline order.
    print("model.act_fun[i].grid",model.act_fun[i].grid.shape) # torch.Size([5, 10])
    # 获取第 i 层 spline 层的 B-spline 系数，这些系数决定了激活函数（B-spline 曲线）的具体形状。
    # coefficients of B-spline bases. (in_features,out_features, grid_size + spline_order)
    print("model.act_fun[i].coef",model.act_fun[i].coef.shape) # torch.Size([5, 1, 6])

    # an activation function phi(x) = sb_scale * b(x) + sp_scale * spline(x)
    # scale_base:mangitude of the base function b(x)
    # scale_base_mu:magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_mu
    # scale_base_sigma: float. magnitude of the residual function b(x) is drawn from N(mu, sigma^2), mu = sigma_base_sigma
    # scale_sp: mangitude of the spline function spline(x)
    base_fun = lambda x: x,  # base_fun是residual function or the base function b(x)
    print("model.act_fun[i].scale_base",model.act_fun[i].scale_base.shape) # torch.Size([5, 1])
    print("model.act_fun[i].scale_sp",model.act_fun[i].scale_sp.shape) # torch.Size([5, 1])


    # 符号层
    # 获取第 i 层符号层中各激活函数的名称（字符串列表），说明当前每个边上使用的符号函数类型，如 'sin'、'x^2' 等。
    # size=(1,in_features),即该层激活函数的数量，边的数量
    print("model.act_fun[i].funs_name",model.symbolic_fun[i].funs_name) # [['0', '0', '0', '0', '0']]
    # mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
    # 获取第 i 层符号层中对应连接的掩码，用于调控各个符号激活函数在前向传播中的贡献（例如剪枝或使某些连接失效）。
    # size=(out_features, in_features)
    print("model.act_fun[i].mask",model.symbolic_fun[i].mask.shape) # torch.Size([1, 5])


# 4 提取KAN中某一个激活函数进行可视化
def extract_func(model,l,i,j):
    '''
    定义要提取的激活函数所在的层索引 l、输入神经元索引 i 和输出神经元索引 j。从模型中提取激活函数的输入值和输出值。
    Args:
        l:
        i:
        j:
    Returns:
    '''
    inputs = model.spline_preacts[l][:, j, i]  # 提取第 l 层，输出神经元索引为 j，输入神经元索引为 i 的激活函数的输入值。
    outputs = model.spline_postacts[l][:, j, i]  # 提取第 l 层，输出神经元索引为 j，输入神经元索引为 i 的激活函数的输出值。
    print("model.get_range(l, i, j)",model.get_range(l, i, j))  # 获取激活函数的范围
    # 由于提取的输入值可能不是有序的，因此需要对输入值进行排序，并根据排序索引重新排列输入值和输出值。
    rank = np.argsort(inputs)  # 获取输入值的排序索引。
    inputs = inputs[rank]  # 根据排序索引重新排列输入值和输出值，确保输入值是有序的。
    outputs = outputs[rank]
    plt.plot(inputs, outputs, marker="o")  # 使用圆形标记每个数据点，以便更清晰地展示输入值和输出值之间的关系。
    plt.title('my_plot')
    plt.show()
l = 0;i = 0;j = 0
# extract_func(model, l, i, j)
# 查看修改前后符号函数有没有变化 - 好像没有变化
model.fix_symbolic(0, 0, 0, 'sin', log_history=False)  # set (l,i,j) activation to be symbolic (specified by fun_name)
extract_func(model, l, i, j)
model.plot()
plt.show()
exit()

# 5 训练模型 -  Train KAN with sparsity regularization
print("train")
model.fit(dataset, opt="LBFGS", steps=10,
          lamb=0.001, # lamb为总体惩罚强度
          lamb_entropy=0.0) # lamb_entropy为熵的（相对）惩罚强度
model.plot() # Plot trained KAN
plt.show()

# 6 修剪 - Prune KAN and replot
# 我们通常使用修剪使神经网络更稀疏，从而更有效和更可解释。kan提供两种修剪方式：自动修剪和手动修剪。
# 对于每个节点，如果其最大传入l1和传出l1都高于某个阈值，我们认为它是活动的（详见论文）。
# 只有活跃的神经元会被保留，而不活跃的神经元会被修剪掉。
# NOTE：没有自动的边修剪，只是为了安全起见（某些情况下，重要的边具有较小的l1范数）,但可以手动修剪掉节点和边。
print("prune")
# 6.1 自动修剪 - both nodes and edges
# 如果一个神经元的属性得分低于给定node_th，则认为该神经元已死亡，并将其设置为零。
# 如果一个边的属性得分低于给定edge_th，则认为其已死亡，并将其设置为零。
model = model.prune(node_th=1e-2, edge_th=3e-2)
# 6.2 pruning nodes
mode = 'auto'
if mode == 'auto':
    # 如果一个神经元的属性得分低于给定阈值，则认为该神经元已死亡，并将被删除
    model = model.prune_node(threshold=1e-2)  # by default the threshold is 1e-2
elif mode == 'manual':
    # 通过传入目标神经元的id进行修剪
    model = model.prune_node(active_neurons_id=[[0]])
# 6.3 修剪边
mode = 'auto'
if mode == 'auto':
    # 自动如果一个边的属性得分低于给定阈值，则认为其已死亡，并将其设置为零。
    model.prune_edge(threshold=3e-2)
elif mode == 'manual':
    # 对特定边进行修剪，即设置其mask to zero
    model.remove_edge(0,0,1)
model.plot() # Plot Pruned KAN
plt.show()

# 7 训练剪枝后的模型 - Continue training and replot
print("训练剪枝后的模型")
model.fit(dataset, opt="LBFGS", steps=10)
model = model.refine(10) # grid refinement？？？？？？？？？？？
model.fit(dataset, opt="LBFGS", steps=10)
model.plot() # Plot 训练的剪枝模型
plt.show()

# 8 设置符号函数并继续训练模型 - Automatically or manually set activation functions to be symbolic
print("设置符号函数然后继续训练模型 - again train")
mode = "auto" # "manual"
if mode == "manual":
    # 边索引（激活函数）
    # 通过layer index、input neuron index、output neuron index手动设置符号函数（即激活函数）
    # 固定符号激活函数，将第 1 层、第 1 个输入和第 1 个输出之间的激活函数固定为符号函数-sin函数，即用符号版本的激活函数替换原来的可学习函数。
    model.fix_symbolic(0,0,0,'sin',log_history=False) # set (l,i,j) activation to be symbolic (specified by fun_name)
    # model.unfix_symbolic(0,0,0) # 取消符号固定，将之前固定为 'sin' 的激活函数恢复为原先的状态（即取消符号固定），使该边上的激活函数重新变为可学习状态或默认设置
    model.fix_symbolic(0,1,0,'x^2',log_history=False)
    model.fix_symbolic(1,0,0,'exp',log_history=False)
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)
    # model.auto_symbolic() # 默认也有lib参数，不设置也行

model.fit(dataset, opt="LBFGS", steps=10)
model.saveckpt('./model') # 保存训练好的模型


# 9 打印出最终拟合的函数
from kan.utils import ex_round
pred_fomula=ex_round(model.symbolic_formula()[0][0], 4) # 将拟合公式的系数取整到小数点后4位
print("pred_fomula",pred_fomula)
model.plot() # Plot final KAN
plt.show()

# 10 模型预测
print("test")
predicted = model(x_tensor).detach().numpy()
print(predicted[:10])
plt.title("MultiKAN")
plt.savefig(r"./figures/InitKAN") # 保存模型结构
plt.show()

# 11 可视化结果
plt.scatter(x, y, label='Original Data')
plt.plot(x, predicted, label=pred_fomula, color='red')
plt.legend()
plt.show()


# 12 加载历史模型
model=KAN.loadckpt('./model') # 加载训练的模型
predicted2 = model(x_tensor).detach().numpy()
print("loadckpt",predicted2[:10])
print(predicted==predicted2)
model.plot() # Plot final KAN
plt.show()