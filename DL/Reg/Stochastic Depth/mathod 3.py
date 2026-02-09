
def residual_drop(x, input_shape, output_shape, strides=(1, 1)):
    global add_tables

    nb_filter = output_shape[0]
    conv = Convolution2D(nb_filter, 3, 3, subsample=strides,
                         border_mode="same", W_regularizer=l2(weight_decay))(x)
    conv = BatchNormalization(axis=1)(conv)
    conv = Activation("relu")(conv)
    conv = Convolution2D(nb_filter, 3, 3,
                         border_mode="same", W_regularizer=l2(weight_decay))(conv)
    conv = BatchNormalization(axis=1)(conv)

    if strides[0] >= 2:
        x = AveragePooling2D(strides)(x)

    if (output_shape[0] - input_shape[0]) > 0:
        pad_shape = (1,
                     output_shape[0] - input_shape[0],
                     output_shape[1],
                     output_shape[2])
        padding = K.zeros(pad_shape)
        padding = K.repeat_elements(padding, K.shape(x)[0], axis=0)
        x = Lambda(lambda y: K.concatenate([y, padding], axis=1),
                   output_shape=output_shape)(x)

    _death_rate = K.variable(death_rate)
    scale = K.ones_like(conv) - _death_rate
    conv = Lambda(lambda c: K.in_test_phase(scale * c, c),
                  output_shape=output_shape)(conv)

    out = merge([conv, x], mode="sum")
    out = Activation("relu")(out)

    gate = K.variable(1, dtype="uint8")
    add_tables += [{"death_rate": _death_rate, "gate": gate}]
    return Lambda(lambda tensors: K.switch(gate, tensors[0], tensors[1]),
                  output_shape=output_shape)([out, x])