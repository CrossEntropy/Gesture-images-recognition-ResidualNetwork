from keras.layers import AveragePooling2D, ZeroPadding2D
from keras.layers import Input, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.models import Model
from keras.utils.vis_utils import plot_model
from Kears_model.conv_block import *
from Kears_model.id_block import *
from tutorial import *
import time
import keras.backend as K
K.set_image_data_format("channels_last")


def ResNets_50(input_shape=(64, 64, 3), classes=6):
    """
    按照下列结构，建立一个50层的残差网络：
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    Arguments:
    input_shape-- shape of the images of the dataset
    classes--  integer, number of classes
    Returns:
    model-- a Model() instance in Keras
    """

    # 定义输入tensor(个人理解成占位符)
    X_input = Input(shape=input_shape)

    # Zero-padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1:
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="valid", name="conv1",
               kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name="conv_bn1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2:
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")

    # Stage 3:
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    # Stage 4:
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    # Stage 5:
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

    # Avearge pooling
    X = AveragePooling2D(pool_size=(2, 2), name="avg_pool")(X)

    # Flatten
    X = Flatten()(X)

    # sotfmax层
    X = Dense(units=classes, activation="softmax", name="fc"+str(classes), kernel_initializer=glorot_uniform(0))(X)

    # 实例化一个Keras model
    model = Model(inputs=X_input, outputs=X, name="ResNets50")

    return model


if __name__ == "__main__":

    # 对signal dataset进行学习和测试
    # 载入数据集
    train_X_orig, train_Y_orig, test_X_orig, test_Y_orig, classes = load_data()
    # 对特征和标签进行处理
    train_X = train_X_orig / 255
    test_X = test_X_orig / 255
    train_Y = convert_to_one_hot(train_Y_orig, 6).T   # 将训练样本的标签转换成one-hot模式，并将标签按行摆放
    test_Y = convert_to_one_hot(test_Y_orig, 6).T     # 将测试样本的标签转换成one-hot模式，并将标签按行摆放

    input_shape = (64, 64, 3)                         # 输入特征的尺寸=(64, 64, 3)
    classes = classes.shape[0]                        # 类别的数量=6
    # 构建一个50层的模型
    model = ResNets_50(input_shape, classes)
    # 对模型进行编译
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # 进行学习
    start = time.time()  # 对训练模型的花费的时间进行评估
    model.fit(x=train_X, y=train_Y, epochs=50, batch_size=32)
    end = time.time()
    print("训练模型耗费的时间: "+str(end-start)+"s")    # 训练模型耗费的时间: 448.1367087364197s

    # 采用测试集评估
    pre = model.evaluate(x=test_X, y=test_Y)
    print("测试集的loss: " + str(pre[0]))
    print("测试集的accuracy: "+ str(pre[1]))
    # 将模型保存
    model.save(filepath=r"C:\python_programme\Andrew_Ng\CLASS_4\week2\Residula_network\修改后的模型\保存好的模型\model_4.h5")
    # 将模型的图画出
    plot_model(model, "model_3.png")


