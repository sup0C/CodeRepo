import torch


def run_rl_search(
        search_space, X_train, y_train, X_val, y_val, num_epochs=10, num_episodes=5
):
    '''
    强化学习的搜索过程采用策略梯度方法，通过最大化期望奖励来优化控制器参数。
    '''
    # 使用ArchitectureController类初始化控制器
    controller = ArchitectureController(search_space)
    controller_optimizer = optim.Adam(controller.parameters(), lr=0.01)

    # 开始搜索
    best_loss = float('inf')
    best_architecture = None
    for episode in range(num_episodes):
        # 梯度置零
        controller_optimizer.zero_grad()

        # rnn期望输入形状为(batch_size, timesteps, features)
        hidden = torch.zeros(1, 1, 64)

        # 初始化列表/字典来存储对数概率和架构选择
        log_probs = []
        architecture = {}

        # 测试架构选择
        for i, key in enumerate(controller.keys):
            # 执行控制器
            logits, hidden = controller(torch.zeros(1, 1, 1), hidden)

            # 为当前架构选择创建分类分布
            dist = torch.distributions.Categorical(logits=logits[i])

            # 从分布中采样一个动作
            action_index = dist.sample()

            # 存储选择的架构值和对数概率
            architecture[key] = search_space[key][action_index.item()]
            log_probs.append(dist.log_prob(action_index))

        # 计算验证损失
        val_loss = evaluate_architecture(architecture, X_train, y_train, X_val, y_val, num_epochs=num_epochs)

        # 更新最优架构选择
        reward = -val_loss
        policy_loss = torch.sum(torch.stack(log_probs) * -reward)
        policy_loss.backward()
        controller_optimizer.step()

        if val_loss < best_loss:
            best_loss = val_loss
            best_architecture = architecture
    return best_architecture, best_loss


best_arch_rl, best_perf_rl = run_rl_search(
    search_space, X_train, y_train, X_val, y_val, num_episodes=5
)