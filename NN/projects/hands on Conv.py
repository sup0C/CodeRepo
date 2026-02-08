def forward(self, x):
    batch_size, in_channels, height, width = x.size()
    assert x.dim() == 4
    assert in_channels == self.in_channels

    # Unfold the input tensor to extract flattened sliding blocks from a batched input tensor.
    # Input:  [batch_size, in_channels, height, width]
    # Output: [batch_size, in_channels*kernel_size*kernel_size, num_patches]
    patches = self.unfold(x)

    # Transpose to have the patches dimension last.
    # Input:  [batch_size, in_channels*kernel_size*kernel_size, num_patches]
    # Output: [batch_size, num_patches, in_channels*kernel_size*kernel_size]
    patches = patches.transpose(1, 2)

    # Reshape the patches to fit the linear layer input requirements.
    # Input:  [batch_size, num_patches, in_channels*kernel_size*kernel_size]
    # Output: [batch_size*num_patches, in_channels*kernel_size*kernel_size]
    patches = patches.reshape(-1, in_channels * self.kernel_size * self.kernel_size)

    # Apply KAN layer to each patch.
    # Input:  [batch_size*num_patches, in_channels*kernel_size*kernel_size]
    # Output: [batch_size*num_patches, out_channels]
    # 这里相当于每个通道的特征图被分为N个patch，每个patch为一个卷积核大小（kernel_size*kernel_size），
    # 然后一共需要训练in_channels*kernel_size*kernel_size乘out_channels个激活函数，
    # 当输出通道数为1时，则只需要训练in_channels*kernel_size*kernel_size个激活函数
    # print("patches.shape",patches.shape)
    out = self.mykan(patches)

    # Reshape the output to the normal format
    # Input:  [batch_size*num_patches, out_channels]
    # Output: [batch_size, num_patches, out_channels]
    out = out.view(batch_size, -1, out.size(-1))

    # Calculate the height and width of the output.
    out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
    out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

    # Transpose back to have the channel dimension in the second position.
    # Input:  [batch_size, num_patches, out_channels]
    # Output: [batch_size, out_channels, num_patches]
    out = out.transpose(1, 2)

    # Reshape the output to the final shape
    # Input:  [batch_size, out_channels, num_patches]
    # Output: [batch_size, out_channels, out_height, out_width]
    out = out.view(batch_size, self.out_channels, out_height, out_width)

    return out