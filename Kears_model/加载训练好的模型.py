from keras.models import load_model
from CLASS_4.week2.Residula_network.tutorial import *

model_1_file = r"C:\python_programme\Andrew_Ng\CLASS_4\week2\Residula_network\修改后的模型\保存好的模型\model_1.h5"
model_2_file = r"C:\python_programme\Andrew_Ng\CLASS_4\week2\Residula_network\修改后的模型\保存好的模型\model_2.h5"
model_3_file = r"C:\python_programme\Andrew_Ng\CLASS_4\week2\Residula_network\修改后的模型\保存好的模型\model_3.h5"

model_1 = load_model(model_1_file)
model_2 = load_model(model_2_file)
model_3 = load_model(model_3_file)

# 对signal dataset进行学习和测试
# 载入数据集
train_X_orig, train_Y_orig, test_X_orig, test_Y_orig, classes = load_data()
# 对特征和标签进行处理
train_X = train_X_orig / 255
test_X = test_X_orig / 255
train_Y = convert_to_one_hot(train_Y_orig, 6).T  # 将训练样本的标签转换成one-hot模式，并将标签按行摆放
test_Y = convert_to_one_hot(test_Y_orig, 6).T  # 将测试样本的标签转换成one-hot模式，并将标签按行摆放
print("-------评估模型--------")
pre = model_1.evaluate(x=train_X, y=train_Y)
print("模型1在训练集的准确率为: "+str(pre[1]))  # 训练集的准确率为: 0.975925925484
pre = model_1.evaluate(x=test_X, y=test_Y)
print("模型1在测试集的准确率为: "+str(pre[1]))  # 测试集的准确率为: 0.941666666667

pre = model_2.evaluate(x=train_X, y=train_Y)
print("模型2在训练集的准确率为: "+str(pre[1]))  # 训练集的准确率为:  0.993518518519
pre = model_2.evaluate(x=test_X, y=test_Y)
print("模型2在测试集的准确率为: "+str(pre[1]))  # 测试集的准确率为:  0.975

pre = model_3.evaluate(x=train_X, y=train_Y)  # 训练集的准确率为:  0.999074074074
print("模型3在训练集的准确率为: "+str(pre[1]))
pre = model_3.evaluate(x=test_X, y=test_Y)    # 测试集的准确率为:  0.950000003974
print("模型3在测试集的准确率为: "+str(pre[1]))