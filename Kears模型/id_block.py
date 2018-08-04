"""
将 Conv2D中的bias去掉
将 权重初始化改为 Glorot or Xaiver_nomal
"""
import tensorflow as tf
from keras.layers import Conv2D, Activation, Add, BatchNormalization
from keras.initializers import  glorot_normal
from keras.regularizers import l2
import numpy as np
import keras.backend as K
K.set_image_data_format('channels_last')


def identity_block(X, f, filters, stage, block, lamd=0.00001, lamd1=0.00001, lamd2=0.00001):
    """
    实施一个恒等块(identity block)

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # 定义层的名字
    conv_base_name = "res" + str(stage) + block + "_branch"
    batch_base_name = "bn" + str(stage) + block + "_branch"

    # 取回滤波器的个数
    F1, F2, F3 = filters

    # 保存输入值，在short cut和main path 的连接中会用到它
    X_shorcut = X

    # 卷积层1, 值得注意的是每个卷积层的后面都要跟着BatchNormalization
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), name=conv_base_name+"2a", padding="valid",
               kernel_initializer=glorot_normal(0), use_bias=False, kernel_regularizer=l2(lamd))(X)
    X = BatchNormalization(axis=3, name=batch_base_name+"2a", gamma_regularizer=l2(lamd1), beta_regularizer=l2(lamd2))(X)
    X = Activation("relu")(X)

    # 卷积层2
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), name=conv_base_name+"2b", padding="same",
               kernel_initializer=glorot_normal(0), use_bias=False, kernel_regularizer=l2(lamd))(X)
    X = BatchNormalization(axis=3, name=batch_base_name+"2b", gamma_regularizer=l2(lamd1), beta_regularizer=l2(lamd2))(X)
    X = Activation("relu")(X)

    # 卷积层3, 不要加激活函数
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), name=conv_base_name+"2c", padding="valid",
               kernel_initializer=glorot_normal(0), use_bias=False, kernel_regularizer=l2(lamd))(X)
    X = BatchNormalization(axis=3, name=batch_base_name+"2c", gamma_regularizer=l2(lamd1), beta_regularizer=l2(lamd2))(X)

    # 将shorcut 加到 main path中，然后再激活
    X = Add()([X_shorcut, X])
    X = Activation("relu")(X)

    return X

if __name__ == "__main__":
    np.random.seed(1)
    A_prev = tf.placeholder(dtype=tf.float32, shape=[3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, 2, [2, 4, 6], stage=1, block="a")
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    A_value = sess.run(A, feed_dict={A_prev: X})
    print(A_value[1][1][0])


