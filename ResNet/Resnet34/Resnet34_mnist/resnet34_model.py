import keras
from keras.layers import Conv2D, BatchNormalization, Activation
from keras import layers, models


class resnet34_model():
    def __init__(self):
        inputs = keras.Input((28,28,1))
        output = self.ResNet34(inputs)
        self.resnet34_model = models.Model(inputs, output)

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

    def ResNet34(self, inputs):
        x = self.ConvCall(inputs, 64, 3, 3, strides=(1, 1))
        x = layers.Activation('relu')(x)

        x = self.ResNetblock(x, 64, strides=(1, 1))
        x = self.ResNetblock(x, 64, strides=(1, 1))
        x = self.ResNetblock(x, 64, strides=(1, 1))

        # 第一个残差块具有下采样功能，步长为2
        x = self.ResNetblock(x, 128, strides=(2, 2))
        x = self.ResNetblock(x, 128, strides=(1, 1))
        x = self.ResNetblock(x, 128, strides=(1, 1))
        x = self.ResNetblock(x, 128, strides=(1, 1))

        x = self.ResNetblock(x, 256, strides=(2, 2))
        x = self.ResNetblock(x, 256, strides=(1, 1))
        x = self.ResNetblock(x, 256, strides=(1, 1))
        x = self.ResNetblock(x, 256, strides=(1, 1))
        x = self.ResNetblock(x, 256, strides=(1, 1))
        x = self.ResNetblock(x, 256, strides=(1, 1))

        x = self.ResNetblock(x, 512, strides=(2, 2))
        x = self.ResNetblock(x, 512, strides=(1, 1))
        x = self.ResNetblock(x, 512, strides=(1, 1))
        x = layers.GlobalAveragePooling2D()(x)  # 全局平均池化
        # output = layers.Dense(1, "softmax")(x)
        output = layers.Dense(10, "softmax")(x)
        return output

    def model_create(self, learning_rate):
        # self.resnet18_model.summary()
        # self.resnet18_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        #                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        #                             metrics=['accuracy'])
        self.resnet34_model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                                    optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
                                    metrics=['accuracy'])
        # self.resnet18_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        #                             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        #                             metrics=['accuracy'])
        return self.resnet34_model
