import sys

import numpy as np

from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.merge import add

sys.setrecursionlimit(10000)


# losses that need sigmoid on top of last layer
yes_softmax = ['crossentropy', 'forward', 'est_forward', 'backward',
               'est_backward', 'boot_soft', 'savage']
# unhinged needs bounded models or it diverges
yes_bound = ['unhinged', 'sigmoid']


def cifar10_resnet(depth, cifar10model, decay, loss):

    model = cifar10model
    input = Input(shape=(model.img_rows, model.img_cols, model.img_channels))

    # 1 conv + BN + relu
    b = Conv2D(filters=16, kernel_size=(model.num_conv, model.num_conv),
               kernel_initializer="he_normal", padding="same",
               kernel_regularizer=l2(decay), bias_regularizer=l2(0))(input)
    b = BatchNormalization(axis=1)(b)
    b = Activation("relu")(b)

    # out = Flatten()(b)
    # dense = Dense(output_dim=model.classes)(out)
    # return Model(inputs=input, outputs=dense)

    # 1 res, no striding
    b = residual(model, decay, first=True)(b)  # 2 layers inside
    for _ in np.arange(1, depth):  # start from 1 => 2 * depth in total
        b = residual(model, decay)(b)

    # 2 res, with striding
    b = residual(model, decay, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = residual(model, decay)(b)

    # 3 res, with striding
    b = residual(model, decay, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = residual(model, decay)(b)

    b = BatchNormalization(axis=1)(b)
    b = Activation("relu")(b)

    b = AveragePooling2D(pool_size=(8, 8), strides=(1, 1),
                         padding="valid")(b)

    out = Flatten()(b)
    if loss in yes_softmax:
        dense = Dense(output_dim=model.classes, kernel_initializer="he_normal",
                      activation="softmax",
                      kernel_regularizer=l2(decay), bias_regularizer=l2(0))(out)
    elif loss in yes_bound:
        dense = Dense(output_dim=model.classes, kernel_initializer="he_normal",
                      kernel_regularizer=l2(decay), bias_regularizer=l2(0))(out)
        dense = BatchNormalization(axis=1)(dense)
    else:
        dense = Dense(output_dim=model.classes, kernel_initializer="he_normal",
                      kernel_regularizer=l2(decay), bias_regularizer=l2(0))(out)

    return Model(inputs=input, outputs=dense)


def residual(model, decay, more_filters=False, first=False):

    def f(input):
        in_channel = input._keras_shape[1]

        if more_filters:
            out_channel = in_channel * 2
            stride = 2
        else:
            out_channel = in_channel
            stride = 1

        if not first:
            b = BatchNormalization(axis=1)(input)
            b = Activation("relu")(b)
        else:
            b = input

        b = Conv2D(filters=out_channel,
                   kernel_size=(model.num_conv, model.num_conv),
                   strides=(stride, stride),
                   kernel_initializer="he_normal", padding="same",
                   kernel_regularizer=l2(decay), bias_regularizer=l2(0))(b)
        b = BatchNormalization(axis=1)(b)
        b = Activation("relu")(b)
        res = Conv2D(filters=out_channel,
                     kernel_size=(model.num_conv, model.num_conv),
                     kernel_initializer="he_normal", padding="same",
                     kernel_regularizer=l2(decay), bias_regularizer=l2(0))(b)

        if more_filters:
            input = Conv2D(filters=out_channel, kernel_size=(1, 1),
                           strides=(2, 2), kernel_initializer="he_normal",
                           padding="valid", kernel_regularizer=l2(decay))(input)

            # input = AveragePooling2D(pool_size=(1, 1), strides=(2, 2),
                                    #  padding="valid")(input)
            # input = ZeroPadding2D(padding=(1, 1))(input)

        return add([input, res])

    return f


def cifar10_plain(depth, cifar10model, decay, loss):

    model = cifar10model
    input = Input(shape=(model.img_channels, model.img_rows, model.img_cols))

    # 1 conv + BN + relu
    b = Conv2D(filters=16, kernel_size=(model.num_conv, model.num_conv),
               kernel_initializer="he_normal", padding="same",
               kernel_regularizer=l2(decay), bias_regularizer=l2(0))(input)
    b = BatchNormalization(axis=1)(b)
    b = Activation("relu")(b)

    # 1 res, no striding
    b = plain(model, decay, first=True)(b)
    for _ in np.arange(1, depth):
        b = plain(model, decay)(b)

    # 2 res, with striding
    b = plain(model, decay, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = plain(model, decay)(b)

    # 3 res, with striding
    b = plain(model, decay, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = plain(model, decay)(b)

    b = BatchNormalization(axis=1)(b)
    b = Activation("relu")(b)

    b = AveragePooling2D(pool_size=(8, 8), strides=(1, 1),
                         padding="valid")(b)

    out = Flatten()(b)
    if loss in yes_softmax:
        dense = Dense(output_dim=model.classes, kernel_initializer="he_normal",
                      activation="softmax",
                      kernel_regularizer=l2(decay), bias_regularizer=l2(0))(out)
    elif loss in yes_bound:
        out = BatchNormalization(axis=1)(out)
        dense = Dense(output_dim=model.classes, kernel_initializer="he_normal",
                      kernel_regularizer=l2(decay), bias_regularizer=l2(0))(out)
    else:
        dense = Dense(output_dim=model.classes, kernel_initializer="he_normal",
                      kernel_regularizer=l2(decay), bias_regularizer=l2(0))(out)

    return Model(inputs=input, outputs=dense)


def plain(model, decay, more_filters=False, first=False):

    def f(input):
        in_channel = input._keras_shape[1]

        if more_filters:
            out_channel = in_channel * 2
            stride = 2
        else:
            out_channel = in_channel
            stride = 1

        if not first:
            b = BatchNormalization(axis=1)(input)
            b = Activation("relu")(b)
        else:
            b = input

        b = Conv2D(filters=out_channel,
                   kernel_size=(model.num_conv, model.num_conv),
                   strides=(stride, stride),
                   kernel_initializer="he_normal", padding="same",
                   kernel_regularizer=l2(decay), bias_regularizer=l2(0))(b)
        b = BatchNormalization(axis=1)(b)
        b = Activation("relu")(b)
        plain = Conv2D(filters=out_channel,
                       kernel_size=(model.num_conv, model.num_conv),
                       kernel_initializer="he_normal", padding="same",
                       kernel_regularizer=l2(decay), bias_regularizer=l2(0))(b)
        return plain

    return f
