import h5py
import numpy as np
import tensorflow as tf


class Config:
    train_path = "E:\\Github_project\\Residual_network\\data_sets\\train_signs.h5"   # 训练集的路径
    test_path = "E:\\Github_project\\Residual_network\\data_sets\\test_signs.h5"     # 测试集的路径
    logdir = "E:\\Github_project\\Residual_network\\TensorFlow_model\\graph"           # event文件存放的路径
    mode_path = "E:\\Github_project\\Residual_network\\TensorFlow_model\\model\\"        # Variable存放的路径
    batch_size = 32    # mini batch的大小
    channels = 3       # 图像的通道数
    height = 64        # 图像的高
    width = 64         # 图像的宽
    paddings = 3       # 填充的圈数
    num_classes = 6    # 类别的总数
    epochs = 50        # epoch的总数
    learning_rate = 0.001      # 学习率


def load_data():
    """"
    加载数据集
    """
    train_dataset = h5py.File(Config.train_path, "r")  # 加载训练集
    train_set_x = np.array(train_dataset["train_set_x"][:])  # 特征
    train_set_y = np.array(train_dataset["train_set_y"][:])  # 标签

    test_dataset = h5py.File(Config.test_path, "r")  # 加载测试集
    test_set_x = np.array(test_dataset["test_set_x"][:])   # 特征
    test_set_y = np.array(test_dataset["test_set_y"][:])   # 标签

    classes = np.array(test_dataset["list_classes"][:])   # 总类别的个数

    return train_set_x, train_set_y, test_set_x, test_set_y, classes


def convert_to_one_hot(Y, classes):
    """
    将标签转化为one_hot模式
    :param Y: np.array, shape is (m, )
    :param classes: int
    :return: np.array, shape is (m, 6)
    """
    Y = np.eye(classes)[Y]
    return Y


def mini_batches(x, y, seed, batch_size=Config.batch_size):
    """
    将数据集shuffled
    :param x: 特征.  np.array, shape is (m, 64, 64, 3)
    :param y: 标签. np.array, shape is (m, )
    :param seed: 随机的种子. int
    :param batch_size: 一个mini_batch的大小
    :return: 划分好的mini_batches. list---->[(x_mini_batch_1, y_min_batch_2), .....]
    """
    np.random.seed(seed)

    # 取得样本的个数
    m = x.shape[0]

    # 将x, y shuffle
    sq = np.random.permutation(m)
    x_shuffle = x[sq]
    y_shuflle = y[sq]

    # 取得mini batch的个数
    num_batches = m // batch_size

    # 进行迭代
    batches = []
    for batch in range(num_batches):
        x_batch = x_shuffle[batch * batch_size: (batch+1) * batch_size]
        y_batch = y_shuflle[batch * batch_size: (batch+1) * batch_size]
        batches.append((x_batch, y_batch))

    return batches


def initialize_weights(shape, fan_in):
    """
    构建权重初始化, 采用的初始化方法为He(何凯明)
    :param shape: 权重的维度。 np.array, shape is (f, f, n_c_prev, n_c)
    :param fan_in: 上一层滤波器的个数, int
    :return: tf.tensor， shape is (f, f, n_c_prev, n_c)
    """
    with tf.name_scope("weights"):
        w = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=np.sqrt(1/fan_in)), name="weights")
        return w


def conv2d(a_prev, filters_h, num_filters, strides, padding):
    """
    构建卷积层
    :param a_prev: 上一层的输出. tf.tensor, shape is (m, n_h_prev, n_w_prev, n_c_prev)
    :param filters_h: 滤波器的尺寸，默认height==width is True. int
    :param num_filters: 滤波器的个数. int
    :param strides: 卷积的步长. int
    :param padding: 卷积的模式. str, "SAME"or"VALID
    :return: 本层的卷积输出. tf.tensor， shape根据padding所定
    """
    # 获得上一层滤波器的个数
    fan_in = a_prev.get_shape().as_list()[-1]

    # 构造下一层权重的维度
    shape = (filters_h, filters_h, fan_in, num_filters)

    # 初始化权重
    w = initialize_weights(shape, fan_in)

    # 进行卷积
    with tf.name_scope("convolution"):
        z = tf.nn.conv2d(a_prev, w, strides=[1, strides, strides, 1], padding=padding, name="z")
    return z


def identity_block(a_prev, num_filters, f=3, training=True):
    """
    构建恒等映射块
    :param a_prev: 上一层的输出. tf.tensor
    :param num_filters: 全部卷积层滤波器的数量. tuple or list, (F1, F2, F3) or [F1, F2, F3]
    :param f: 第二个卷积层滤波器的尺寸. int
    :param training: 训练还是测试 bool
    :return: tf.tensor
    """
    # 滤波器的数量
    F1, F2, F3 = num_filters

    # 构建main path
    with tf.name_scope("Main_path"):
        with tf.name_scope("Conv_1"):    # 第一个卷积层
            z_1 = conv2d(a_prev, filters_h=1, num_filters=F1, strides=1, padding="VALID")
        with tf.name_scope("BN_1"):      # 第一个BN
            z_bn = tf.layers.batch_normalization(z_1, training=training)
        with tf.name_scope("Relu_1"):    # 第一个激活层
            a_1 = tf.nn.relu(z_bn)
        with tf.name_scope("Conv_2"):    # 第二个卷积层
            z_2 = conv2d(a_1, filters_h=f, num_filters=F2, strides=1, padding="SAME")
        with tf.name_scope("BN_2"):      # 第二个BN
            z_bn = tf.layers.batch_normalization(z_2, training=training)
        with tf.name_scope("Relu_2"):    # 第一个激活层
            a_2 = tf.nn.relu(z_bn)
        with tf.name_scope("Conv_3"):    # 第三个卷积层
            z_3 = conv2d(a_2, filters_h=1, num_filters=F3, strides=1, padding="VALID")
        with tf.name_scope("BN_3"):     # 第三个BN
            z_bn = tf.layers.batch_normalization(z_3, training=training)

    # 构建 shortcut, 原文中称之为skip connection
    with tf.name_scope("Shortcut"):
        shortcut = tf.identity(a_prev)

    # 第三个激活层激活层
    with tf.name_scope("Relu_3"):
        a = tf.nn.relu(tf.add(shortcut, z_bn))

    return a


def convolution_block(a_prev, num_filters, f=3, s=2, training=True):
    """
    构建卷积映射块
    :param a_prev: 上一层的输出. tf.tensor
    :param num_filters: 全部卷积层滤波器的数量. tuple or list, (F1, F2，F3) or [F1, F2, F3]
    :param f: 第二个卷积层的滤波器的尺寸. intF
    :param s: 第一个卷积层滤波器的步长. int
    :param training: 训练还是测试. bool
    :return: tf.tensor
    """
    # 滤波器的数量
    F1, F2, F3 = num_filters

    # 构建main path
    with tf.name_scope("Main_path"):
        with tf.name_scope("Conv_1"):   # 第一个卷积层
            z_1 = conv2d(a_prev, filters_h=1, num_filters=F1, strides=s, padding="VALID")
        with tf.name_scope("BN_1"):     # 第一个BN
            z_bn = tf.layers.batch_normalization(z_1, training=training)
        with tf.name_scope("Relu_1"):   # 激活函数
            a_1 = tf.nn.relu(z_bn)
        with tf.name_scope("Conv_2"):   # 第二个卷积层
            z_2 = conv2d(a_1, filters_h=f, num_filters=F2, strides=1, padding="SAME")
        with tf.name_scope("BN_2"):     # 第二个BN
            z_bn = tf.layers.batch_normalization(z_2, training=training)
        with tf.name_scope("Relu_2"):   # 激活函数
            a_3 = tf.nn.relu(z_bn)
        with tf.name_scope("Conv_3"):   # 第三个卷积层
            z_3 = conv2d(a_3, filters_h=1, num_filters=F3, strides=1, padding="VALID")
        with tf.name_scope("BN_3"):   # 第三个BN
            z_bn = tf.layers.batch_normalization(z_3, training=training)

    # 构建 shortcut, 原文中称之为skip connection
    with tf.name_scope("Shortcut"):
        with tf.name_scope("Conv"):
            z_4 = conv2d(a_prev, filters_h=1, num_filters=F3, strides=s, padding="VALID")
        with tf.name_scope("BN"):
            shortcut = tf.layers.batch_normalization(z_4, training=training)

    # 第三个激活层
    with tf.name_scope("Relu_3"):
        a = tf.nn.relu(tf.add(shortcut, z_bn))

    return a


def fully_connected(a_prev, fan_out=6):
    """
    构建 Z = XW + b
    :param a_prev: 上一层的输出. tf.tensor, shape is (batch_size, fan_in)
    :param fan_out: 扇入or本层的神经元的数目. int
    :return: Z. tf.tensor, shape is (batch_size, fan_out)
    """
    # 获得 fan_in
    fan_in = a_prev.get_shape().as_list()[-1]

    # 初始化权重
    w = initialize_weights((fan_in, fan_out), fan_in)

    # 初始化偏置
    with tf.name_scope("biases"):
        b = tf.Variable(tf.zeros(shape=(fan_out, )), dtype=tf.float32)

    # 加权和
    with tf.name_scope("linear_combination"):
        z = tf.nn.bias_add(tf.matmul(a_prev, w), b)

    return z


def flatten(a_prev):
    a = tf.reshape(a_prev, shape=(Config.batch_size, -1))
    return a


if __name__ == "__main__":
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data()
    print(train_x_orig.shape)   # 训练集特征的shape (1080, 64, 64, 3)
    print(train_y_orig.shape)   # 训练集标签的shape (1080,)
    print(test_x_orig.shape)    # 测试集的特征的shape (120, 64, 64, 3)
    print(test_y_orig.shape)    # 测试集的标签的shape (120, )
    print(classes)              # 类别的种类共6类
