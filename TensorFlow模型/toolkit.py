import h5py
import numpy as np
import tensorflow as tf


train_path = r"E:\Github_project\Residual_network\data_sets\train_signs.h5"
test_path = r"E:\Github_project\Residual_network\data_sets\test_signs.h5"


def load_data():
    train_dataset = h5py.File(train_path, "r")  # 加载训练集
    train_set_x = np.array(train_dataset["train_set_x"][:])  # 特征
    train_set_y = np.array(train_dataset["train_set_y"][:])  # 标签

    test_dataset = h5py.File(test_path, "r")  # 加载测试集
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


def initialize_weights(shape, fan_in):
    """
    构建权重初始化, 采用的初始化方法为He
    :param shape: 权重的维度, np.alist, shape is (f, f, n_c_prev, n_c)
    :param fan_in: 上一层神经元的个数, int
    :return: tf.tensor， shape is (f, f, n_c_prev, n_c)
    """
    with tf.name_scope("weights"):
        w = tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=np.sqrt(1/fan_in)), name="weights")
        return w


def conv2d(a_prev, shape, strides, padding):
    """
    构建卷积层
    :param a_prev: 上一层的输出, tf.tensor, shape is (m, n_h_prev, n_w_prev, n_c_prev)
    :param shape: 本层权重的维度, (f, f, n_c_prev, n_c)
    :param strides: 卷积的步长, int
    :param padding: 卷积的模式, str, "SAME"or"VALID
    :return: 本层的卷积输出, tf.tensor， shape根据padding所定
    """
    # 获得上一层神经元的个数
    fan_in = a_prev.get_shape.as_list[-1]

    # 初始化权重
    w = initialize_weights(shape, fan_in)

    # 进行卷积
    with tf.name_scope("convolution"):
        z = tf.nn.conv2d(a_prev, w, strides=strides, padding=padding, name="z")
    return z


def activation(z, act=tf.nn.relu):
    """
    构建激活函数
    :param z: 未经过激活函数的张量, tf.tensor
    :param act: 激活函数的选择, 函数对象
    :return: 经过激活函数的张量
    """
    a = act(z)
    return a






if __name__ == "__main__":
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data()
    print(train_x_orig.shape)   # 训练集特征的shape (1080, 64, 64, 3)
    print(train_y_orig.shape)   # 训练集标签的shape (1080,)
    print(test_x_orig.shape)    # 测试集的特征的shape (120, 64, 64, 3)
    print(test_y_orig.shape)    # 测试集的标签的shape (120, )
    print(classes)              # 类别的种类共6类
    print(convert_to_one_hot(train_y_orig, 6).shape)
