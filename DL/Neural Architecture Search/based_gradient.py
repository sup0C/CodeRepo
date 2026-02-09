import torch.nn as nn
import torch.optim as optim

def run_gradient_based_search(search_space, X_train, y_train, X_val, y_val, num_epochs=50):
    '''
    基于梯度的方法通过构建可微分的搜索空间来直接优化架构参数。
    我们实现了一个简化版的DARTS框架，展示其核心思想。
    DARTS采用交替优化策略，分别更新网络权重和架构参数
    '''
    # 定义模型、损失函数和优化器
    model = Model(search_space)
    criterion = nn.MSELoss()

    arch_params = [model.alphas]
    optimizer_alpha = optim.Adam(arch_params, lr=0.001)
    arch_param_ids = {id(p) for p in arch_params}
    weight_params = [p for p in model.parameters() if p.requires_grad and id(p) not in arch_param_ids]
    optimizer_w = optim.Adam(weight_params, lr=0.01)

    # 开始搜索
    for epoch in range(num_epochs):
        # 梯度置零
        optimizer_w.zero_grad()

        # 前向传播
        outputs = model(X_train)

        # 优化
        loss_w = criterion(outputs, y_train)
        loss_w.backward()
        optimizer_w.step()

        # 反向传播
        optimizer_alpha.zero_grad()
        val_outputs = model(X_val)
        loss_alpha = criterion(val_outputs, y_val)
        loss_alpha.backward()
        optimizer_alpha.step()

    best_architecture = model.discretize()
    final_loss = evaluate_architecture(best_architecture, X_train, y_train, X_val, y_val, num_epochs=50)
    return best_architecture, final_loss


best_arch_gb, best_perf_gb = run_gradient_based_search(
    search_space, X_train, y_train, X_val, y_val)