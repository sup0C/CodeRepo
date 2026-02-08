def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):

    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    # 保留该层的概率
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors（ 2D ConvNets）
    # shape只保留batch_size的值，其它所有维度的值都为1
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # 实现方法1
    # torch.rand从均匀分布[0, 1]中采样，
    # 与keep_prob相加得到random_tensor后最后再向下取整.floor()，
    # 举例
        # 假设drop_prob=0.2即有0.2的概率丢弃该层，
        # 则有keep_prob=1-drop_prob=0.8的概率保留该层，
        # 与[0, 1]均匀分布相加后有0.8的概率大于1向下取整后为1，
        # 0.2的概率小于1向下取整后为0。
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # x.div(keep_prob)和dropout中的操作一样，
    # 因为训练时随机丢弃部分连接或层，推理时不丢弃，除以keep_prob是为了保持总体期望不变。
    output = x.div(keep_prob) * random_tensor.floor()

    # 实现方法2
    # random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    # if keep_prob > 0.0 and scale_by_keep:
    #     random_tensor.div_(keep_prob)
    # output=x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        '''
        存储网络结构的失活概率
        '''
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

