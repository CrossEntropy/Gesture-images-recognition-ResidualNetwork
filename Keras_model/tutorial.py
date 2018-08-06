import h5py
import numpy as np

train_path = r"E:\Github_project\Residual_network\data_sets\train_signs.h5"
test_path = r"E\Github_project\Residual_network\data_sets\est_signs.h5"


def load_data():
    train_dataset = h5py.File(train_path, "r")  # 加载训练集
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 特征
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 标签

    test_dataset = h5py.File(test_path, "r")  # 加载测试集
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])   # 特征
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])   # 标签

    classes = np.array(test_dataset["list_classes"][:])   # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))   # Andrew Ng 在课程中将样本的标签按列摆放
    test_set_y_org = np.reshape(test_set_y_orig, newshape=(1, test_set_y_orig.shape[0]))

    return  train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_org, classes


# 将标签转化为one_hot模式
def convert_to_one_hot(Y, classes):
    Y = np.eye(classes)[Y.reshape(-1)].T
    return Y


if __name__ == "__main__":
    train_x_orig, train_y_orig, test_x_orig, test_y_orig, classes = load_data()
    print(train_x_orig.shape)   # 训练集特征的shape (1080, 64, 64, 3)
    print(train_y_orig.shape)   # 训练集标签的shape (1, 1080)
    print(test_x_orig.shape)    # 测试集的特征的shape (120, 64, 64, 3)
    print(test_y_orig.shape)    # 测试集的标签的shape (1, 120)
    print(classes)              # 类别的种类共6类
