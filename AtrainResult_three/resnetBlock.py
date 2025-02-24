import tensorflow as tf
from tensorflow.keras import layers, models, Input

def conv_block(inputs, filters, kernel_size=3, stride=1, activation='relu'):
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation(activation)(x)
    return x

def identity_block(inputs, filters, kernel_size=3):
    x = conv_block(inputs, filters, kernel_size)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, inputs])
    x = layers.Activation('relu')(x)
    return x

def resnet_block(inputs, filters, kernel_size=3, stride=1):
    x = conv_block(inputs, filters, kernel_size, stride)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv1D(filters, 1, strides=stride)(inputs)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet18(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = conv_block(inputs, 64, kernel_size=7, stride=2)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = resnet_block(x, 64, stride=1)
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    x = identity_block(x, 64)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x)
    return model



