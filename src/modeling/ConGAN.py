import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

def make_generator_model(hidden_layer_num = 3, hidden_layer_strides1_num = 1, basic_filter_num = 64, init_dim = 7,
                         filter_size = 4, norm = 'batch', act = 'leakyrelu'):
    """
    this code is copied from https://www.tensorflow.org/tutorials/generative/dcgan
    make no change
    :return:
    """

    # 4*4*256 -> 8*8*128 -> 16*16*64 -> 32*32*1
    filter_num = basic_filter_num
    layer_filter = [filter_num]
    for i in range(hidden_layer_num-1):
        filter_num = filter_num*2
        layer_filter.append(filter_num)

    if norm == 'batch':
        norm_layer = layers.BatchNormalization

    if act == 'leakyrelu':
        act_layer = layers.LeakyReLU
    elif act == 'relu':
        act_layer = layers.ReLU

    model = tf.keras.Sequential()
    # project
    filter_num_pro = layer_filter.pop()
    model.add(layers.Dense(init_dim*init_dim*filter_num_pro, use_bias=False, input_shape=(100,)))
    if norm != None:
        model.add(norm_layer())
    if act != None:
        model.add(act_layer())
    model.add(layers.Reshape((init_dim, init_dim, filter_num_pro)))
    hidden_layer_num -= 1

    # stride = 1
    for _ in range(hidden_layer_strides1_num):
        filter_num_strides1 = layer_filter.pop()
        model.add(layers.Conv2DTranspose(filter_num_strides1, (filter_size, filter_size), strides=(1, 1),
                                         padding='same', use_bias=False))
        if norm != None:
            model.add(norm_layer())
        if act != None:
            model.add(act_layer())
        hidden_layer_num -= 1

    # upsampling (stride = 2)
    while hidden_layer_num > 0:
        filter_num_strides2 = layer_filter.pop()
        model.add(layers.Conv2DTranspose(filter_num_strides2, (filter_size, filter_size), strides=(2, 2),
                                         padding='same', use_bias=False))
        if norm != None:
            model.add(norm_layer())
        if act != None:
            model.add(act_layer())
        hidden_layer_num -= 1

    model.add(layers.Conv2DTranspose(1, (filter_size, filter_size), strides=(2, 2), padding='same',
                                     use_bias=False, activation='tanh'))

    return model


def make_discriminator_model(img_dim = 32, hidden_layer_num = 3, basic_filter_num = 64, filter_size = 4,
                         norm = 'batch', act = 'leakyrelu', dropout_ratio = 0.3, last_layer = 'dense',
                             if_sigmoid = False):
    """
    this code is copied from https://www.tensorflow.org/tutorials/generative/dcgan
    make no change
    :return:
    """

    if norm == 'batch':
        norm_layer = layers.BatchNormalization
    elif norm == 'layer':
        norm_layer = layers.LayerNormalization

    if act == 'leakyrelu':
        act_layer = layers.LeakyReLU
    elif act == 'relu':
        act_layer = layers.ReLU

    filter_num = basic_filter_num
    layer_filter = [filter_num]
    for i in range(hidden_layer_num):
        filter_num = filter_num*2
        layer_filter.append(filter_num)


    model = tf.keras.Sequential()

    filter_num_strides1 = layer_filter.pop(0)
    model.add(layers.Conv2D(filter_num_strides1, (filter_size, filter_size), strides=(2, 2),
                            padding='same', input_shape=[img_dim, img_dim, 1]))
    if norm != None:
        model.add(norm_layer())
    if act != None:
        model.add(act_layer())
    model.add(layers.Dropout(dropout_ratio))
    hidden_layer_num -= 1

    # downsampling (stride = 2)
    while hidden_layer_num > 0:
        filter_num_strides2 = layer_filter.pop(0)
        model.add(layers.Conv2D(filter_num_strides2, (filter_size, filter_size), strides=(2, 2),
                                         padding='same'))
        if norm != None:
            model.add(norm_layer())
        if act != None:
            model.add(act_layer())
        model.add(layers.Dropout(dropout_ratio))
        hidden_layer_num -= 1

    if last_layer == 'dense':
        model.add(layers.Flatten())
        if norm != None:
            model.add(norm_layer())

        if if_sigmoid:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(1))
    elif last_layer == 'con2d':
        if if_sigmoid:
            model.add(layers.Conv2D(1, (filter_size, filter_size), strides=(1, 1), padding='valid', activation='sigmoid'))
        else:
            model.add(layers.Conv2D(1, (filter_size, filter_size), strides=(1, 1), padding='valid'))
        model.add(layers.Flatten())

    return model





