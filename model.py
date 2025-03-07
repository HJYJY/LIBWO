import keras
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation
from keras import layers, models


class resnet18_model():
    def __init__(self):
        inputs = keras.Input((32, 32, 3))
        output = self.ResNet18(inputs)
        self.resnet18_model = models.Model(inputs, output)

    def ConvCall(self, x, channel, xx, yy, strides=(1, 1)):
        # （x）这是函数调用操作，意味着我们将使用创建的卷积层对输入的张量x进行卷积操作，并将结果赋值给x。
        x = Conv2D(channel, (xx, yy), strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        return x

    def ResNetblock(self, input, channel, strides=(1, 1)):
        x = self.ConvCall(input, channel, 3, 3, strides=strides)
        x = Activation("relu")(x)

        x = self.ConvCall(x, channel, 3, 3, strides=(1, 1))
        if strides != (1, 1):
            residual = self.ConvCall(input, channel, 1, 1, strides=strides)
        else:
            residual = input
        x = x + residual
        x = Activation("relu")(x)

        return x

    def ResNet18(self, inputs):
        x = self.ConvCall(inputs, 32, 3, 3, strides=(1, 1))
        x = layers.Activation('relu')(x)

        # x = self.ResNetblock(x, 64, strides=(1, 1))
        # x = self.ResNetblock(x, 64, strides=(1, 1))
        #
        # x = self.ResNetblock(x, 128, strides=(2, 2))
        # x = self.ResNetblock(x, 128, strides=(1, 1))
        #
        # x = self.ResNetblock(x, 256, strides=(2, 2))
        # x = self.ResNetblock(x, 256, strides=(1, 1))
        #
        # x = self.ResNetblock(x, 512, strides=(2, 2))
        # x = self.ResNetblock(x, 512, strides=(1, 1))
        x = self.ResNetblock(x, 32, strides=(1, 1))
        x = self.ResNetblock(x, 32, strides=(1, 1))

        x = self.ResNetblock(x, 64, strides=(2, 2))
        x = self.ResNetblock(x, 64, strides=(1, 1))

        x = self.ResNetblock(x, 128, strides=(2, 2))
        x = self.ResNetblock(x, 128, strides=(1, 1))

        x = self.ResNetblock(x, 256, strides=(2, 2))
        x = self.ResNetblock(x, 256, strides=(1, 1))
        x = layers.GlobalAveragePooling2D()(x)  # 全局平均池化
        # output = layers.Dense(1, "softmax")(x)
        output = layers.Dense(10, "softmax")(x)
        return output

    def model_create(self, learning_rate):
        self.resnet18_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                                    # optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
                                    optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
                                    metrics=['accuracy'])
        return self.resnet18_model
