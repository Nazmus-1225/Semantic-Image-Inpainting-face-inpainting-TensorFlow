import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
BATCHSIZE = 32
IMG_H = 64
IMG_W = 64
IMG_C = 3
Z_DIM = 100
MASK_H = 32
MASK_W = 32
LAMBDA = 3e-3
IS_DCGAN_TRAINED = False
EPSILON = 1e-14
