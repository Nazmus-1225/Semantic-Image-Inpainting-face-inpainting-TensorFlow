from ops import *
from CONFIG import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, Z):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = tf.reshape(tf.nn.relu((fully_connected("linear", Z, 4*4*512))), [-1, 4, 4, 512])
            inputs = tf.nn.relu(InstanceNorm(uconv("deconv1", inputs, 256, 5, 2), "IN1"))
            inputs = tf.nn.relu(InstanceNorm(uconv("deconv2", inputs, 128, 5, 2), "IN2"))
            inputs = tf.nn.relu(InstanceNorm(uconv("deconv3", inputs, 64, 5, 2), "IN3"))
            inputs = tf.nn.tanh(uconv("deconv4", inputs, IMG_C, 5, 2))
            return inputs

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

class Discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = leaky_relu(conv("conv1", inputs, 64, 5, 2, is_SN=True))
            print(inputs.shape)
            inputs = leaky_relu(InstanceNorm(conv("conv2", inputs, 128, 5, 2, is_SN=True), "IN2"))
            print(inputs.shape)
            inputs = leaky_relu(InstanceNorm(conv("conv3", inputs, 256, 5, 2, is_SN=True), "IN3"))
            print(inputs.shape)
            inputs = leaky_relu(InstanceNorm(conv("conv4", inputs, 512, 5, 2, is_SN=True), "IN4"))
            print(inputs.shape)
            print(inputs.dtype)
            inputs = tf.layers.flatten(inputs)
            return fully_connected("liner", inputs, 1)

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
