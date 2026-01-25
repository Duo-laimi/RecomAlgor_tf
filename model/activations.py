import tensorflow as tf
from tensorflow.keras import layers

class Dice(layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        super(Dice, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        # 这里的 alpha 是可学习的缩放参数
        self.alpha = self.add_weight(
            name='alpha',
            shape=(input_shape[self.axis],),
            initializer='zeros',
            trainable=True
        )
        # 实例化一个 BatchNorm 层来获取均值和方差
        self.bn = layers.BatchNormalization(
            axis=self.axis,
            epsilon=self.epsilon,
            center=False,
            scale=False
        )
        super(Dice, self).build(input_shape)

    def call(self, inputs, training=None):
        # 1. 对输入进行标准化
        inputs_normed = self.bn(inputs, training=training)

        # 2. 计算控制函数 p(s)
        x_p = tf.sigmoid(inputs_normed)

        # 3. 组合输出: p(s) * s + (1 - p(s)) * alpha * s
        output = x_p * inputs + (1 - x_p) * self.alpha * inputs
        return output

    def get_config(self):
        config = super(Dice, self).get_config()
        config.update({"axis": self.axis, "epsilon": self.epsilon})
        return config


ACT_FUNC = {
    # 标准字符串激活（直接调用 Keras 内部映射）
    "relu": lambda: layers.Activation("relu"),
    "sigmoid": lambda: layers.Activation("sigmoid"),
    "tanh": lambda: layers.Activation("tanh"),
    "gelu": lambda: layers.Activation("gelu"),
    "swish": lambda: layers.Activation("swish"),

    # 层型激活函数（需要实例化对象）
    "leaky_relu": lambda: layers.LeakyReLU(alpha=0.2),
    "prelu": lambda: layers.PReLU(),
    "dice": lambda: Dice(),  # 使用我们自定义的 Dice 类
}