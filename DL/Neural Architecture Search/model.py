import torch.nn as nn


# 定义单元
class Cell(nn.Module):
    def __init__(self, in_features, out_features, ops):
        super(Cell, self).__init__()
        self.ops = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, out_features), op()) for op in ops
        ])

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))


# 定义目标模型
class Model(nn.Module):
    def __init__(self, search_space):
        super(Model, self).__init__()
        self.ops_list = [activation_map[name] for name in search_space['activation_function']]
        self.num_ops = len(self.ops_list)
        self.num_hidden_layers = max(search_space['num_hidden_layers'])
        self.hidden_layer_size = search_space['hidden_layer_size'][0]
        self.alphas = nn.Parameter(torch.randn(self.num_hidden_layers, self.num_ops, requires_grad=True))
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(1, self.hidden_layer_size))
        for _ in range(self.num_hidden_layers - 1):
            self.layers.append(Cell(self.hidden_layer_size, self.hidden_layer_size, self.ops_list))
        self.output_layer = nn.Linear(self.hidden_layer_size, 1)

    def forward(self, x):
        architecture_weights = nn.functional.softmax(self.alphas, dim=-1)
        output = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                output = layer(output)
            elif isinstance(layer, Cell):
                output = layer(output, architecture_weights[i - 1])
        return self.output_layer(output)

    def discretize(self):
        architecture = {
            'num_hidden_layers': self.num_hidden_layers,
            'hidden_layer_size': self.hidden_layer_size,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'dropout_rate': 0.0
        }
        best_op_indices = self.alphas.argmax(dim=-1)
        best_ops = [self.ops_list[i].__name__ for i in best_op_indices]
        architecture['activation_function'] = best_ops[0]
        return architecture