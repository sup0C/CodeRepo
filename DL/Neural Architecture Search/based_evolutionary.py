import random
from copy import deepcopy


def run_evolutionary_search(X, y, search_space, population_size=10, num_generations=5):
    '''
    进化算法通过维护一个架构群体并模拟生物进化过程来搜索最优解。
    该方法在每一代中评估群体中所有个体的适应度，然后通过选择、交叉和变异操作产生下一代群体。
    '''
    best_loss = float('inf')
    best_architecture = None

    # 创建训练和验证数据集
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 在群体中开始搜索
    population = []
    for _ in range(population_size):
        # 随机选择要测试的架构
        architecture = {key: random.choice(search_space[key]) for key in search_space}
        population.append(architecture)

    # 迭代所有代（架构选项集合）
    for _ in range(num_generations):
        fitness = []
        for arch in population:
            loss = evaluate_architecture(arch, X_train, y_train, X_val, y_val, num_epochs=10)
            fitness.append((loss, arch))

            if loss < best_loss:
                best_loss = loss
                best_architecture = arch

        # 通过从代中选择'精英'（高性能架构）创建新群体
        fitness.sort(key=lambda x: x[0])
        new_population = []
        num_elites = population_size // 2
        elites = [arch for loss, arch in fitness[:num_elites]]
        new_population.extend(elites)

        # 从新群体创建和变异后代
        while len(new_population) < population_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)

            child = deepcopy({})
            for key in parent1: child[key] = random.choice([parent1[key], parent2[key]])
            mutation_key = random.choice(list(search_space.keys()))
            child[mutation_key] = random.choice(search_space[mutation_key])
            new_population.append(child)

        population = new_population
    return best_architecture, best_loss


best_arch_ea, best_perf_ea = run_evolutionary_search(
    search_space, population_size=10, num_generations=5
)