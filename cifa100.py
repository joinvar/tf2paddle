# 引入要用到的包
import numpy as np
import cv2

# 加载cifar数据集
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

meta = unpickle("./cifar-100-python/meta")
# meta.keys()

# 加载训练集
trainset = unpickle("./cifar-100-python/train")
# trainset.keys()

# 标签
trainset_y = trainset[b'coarse_labels']
# print(trainset_y)
# 训练数据
trainset_x = trainset[b'data']

# 训练集数量
n_trainset = len(trainset_x)
# n_trainset = 64  # 测试用
# 分类数量
n_class = len(meta[b'coarse_label_names'])






# 加载测试集
testset = unpickle("./cifar-100-python/train")
# 标签
testset_y = testset[b'coarse_labels']
# 数据
testset_x = testset[b'data']
# 转换训练集图像
trainset_x = trainset_x.reshape(-1,3,32,32)
trainset_x = np.rollaxis(trainset_x,1,4)
# 转换测试集图像
testset_x = testset_x.reshape(-1,3,32,32)
testset_x = np.rollaxis(testset_x,1,4)



def CLAHE(img):
    # 对比度受限自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    # 将clahe应用到输入图像上
    cl1 = clahe.apply(img)
    return cl1

def Histograms_Equalization(img):
    # 直方图均衡化
    equ = cv2.equalizeHist(img)
    return equ

# one-hot 编码
def make_one_hot(data, num):
    return (np.arange(num) == data[:, None]).astype(np.integer)

# 对训练集做处理

from sklearn.utils import shuffle
import sys
from sklearn.model_selection import train_test_split
new_X_train = []
new_y_train = []
# 循环遍历数据集
for index in range(n_trainset):
    print(index)
    sys.stdout.write("{} / {}\r".format(index, n_trainset))

    # 训练集图像转灰度图
    img_gray = cv2.cvtColor(trainset_x[index], cv2.COLOR_RGB2GRAY)
    # 保存归一化图像和目标y(分类ID)
    new_X_train.append(img_gray.astype('float32') / 255.0)
    new_y_train.append(trainset_y[index])
    # 直方图归一化并保存归一化图像和目标y(分类ID)
    he_img = Histograms_Equalization(img_gray)
    new_X_train.append(he_img.astype('float') / 255.0)
    new_y_train.append(trainset_y[index])
    # CLAHE并保存归一化图像和目标(分类ID)
    clane_img = CLAHE(img_gray)
    new_X_train.append(clane_img.astype('float32') / 255.0)
    new_y_train.append(trainset_y[index])
print("All done!")
# 将输入数据扩展一个维度，(n, 32, 32) 变成(n, 32, 32, 1)
all_xs = np.expand_dims(new_X_train, 3)
# 将目标y转成one-hot编码
all_ys = make_one_hot(np.array(new_y_train), n_class)
# 随机打散训练集，并按照比例将训练集拆分成训练集和验证集
train_xs, valid_xs, train_ys, valid_ys = train_test_split(all_xs, all_ys,
test_size = 0.2, random_state = 0)
# print("aaaaa  train_ys shape ",train_xs.shape,"fddd",train_ys.shape)





# 处理测试集
from sklearn.utils import shuffle
test_set = []
# 循环 遍历测试集
for j in range(len(testset_x)):
    # 图像转灰度图
    img_gray = cv2.cvtColor(testset_x[j], cv2.COLOR_RGB2GRAY)
    # 扩展维度，(32, 32) 转成 (32, 32, 1)
    # img_gray = np.expand_dims(img_gray, 2)
    # 扩展维度，(32, 32) 转成 (1, 32, 32)
    img_gray = np.expand_dims(img_gray, 0)
    # 归一化
    img_gray = img_gray / 255.0
    # 保存图像
    test_set.append(img_gray)

test_set = np.array(test_set)
y_test = make_one_hot(np.array(testset_y), n_class)

import paddle.fluid as fluid
import paddle
import datetime


# 定义网络模型
def convolutional_neural_network(input_img):
    # 卷积层：输入=32*32*1,输出=28*28*100，激活函数ReLu
    conv1 = fluid.layers.conv2d(input_img, num_filters=100, filter_size=5, act="relu", data_format="NHWC")
    print("conv1.shape = ", conv1.shape)  # (?,28,28,100)
    pool1 = fluid.layers.pool2d(conv1, pool_size=2, pool_stride=2, pool_type="max", data_format="NHWC")
    print("pool1.shape = ", pool1.shape)  # (?,14,14,100)

    conv2 = fluid.layers.conv2d(pool1, num_filters=150, filter_size=3, act="relu", data_format="NHWC")
    print("conv2.shape = ", conv2.shape)  # (?, 12, 12, 150)
    pool2 = fluid.layers.pool2d(conv2, pool_size=2, pool_stride=2, pool_type="max", data_format="NHWC")
    print("pool2.shape = ", pool2.shape)  # (?, 6, 6, 150)

    conv3 = fluid.layers.conv2d(pool2, num_filters=250, filter_size=3, padding="same", act="relu", data_format="NHWC")
    print("conv3.shape = ", conv3.shape)  # (?, 6, 6, 250)
    pool3 = fluid.layers.pool2d(conv3, pool_size=2, pool_stride=2, pool_type="max", data_format="NHWC")
    print("pool3.shape = ", pool3.shape)  # (?, 3, 3, 250)

    fc0 = fluid.layers.flatten(pool3)
    print("fc0.shape = ", fc0.shape)  # (?, 2250)

    fc1 = fluid.layers.fc(fc0, size=512, act='relu')
    print("fc1.shape = ", fc1.shape)  # (?, 300)
    fc2 = fluid.layers.fc(fc1, size=300, act='relu')

    logits = fluid.layers.fc(fc2, size=n_class)
    print("logits.shape = ", logits.shape)  # (?, 20)
    return logits


# 交叉熵误差
def cross_entropy(predict, labels):
    print("labels = ", labels.shape)
    cross_entropy = fluid.layers.reduce_mean(
        fluid.layers.softmax_with_cross_entropy(predict, labels, soft_label=True))

    return cross_entropy


# 优化器
def optimizer(cross_entropy):
    optimizer = fluid.optimizer.AdamOptimizer(1e-4)
    train_step = optimizer.minimize(cross_entropy)
    return train_step


# 计算准确率
def accurc(predict, laberls):
    correct_prediction = fluid.layers.argmax(predict, 1) == fluid.layers.argmax(laberls, 1)
    accuracy = fluid.layers.reduce_mean(fluid.layers.cast(correct_prediction, dtype='float32'))
    return accuracy


def train_program():
    # 定义输入占位符
    images = fluid.data('input_tensor', shape=[-1, 32, 32, 1], dtype='float32')
    print(images.shape)
    predict = convolutional_neural_network(images)

    # 定义目标y占位符
    label = fluid.data('labels', shape=[-1, n_class], dtype='float32')
    avg_cost = cross_entropy(predict, label)
    opt = optimizer(avg_cost)
    accuracy = accurc(predict, label)
    return [avg_cost, opt, accuracy]


# 模型保存路径
param_base_dir = '/home/aistudio/param'
import os

infer_param_path = os.path.join(param_base_dir, "cst_inf")  # 推理裁剪过的模型保存路径
ckpt_param_path = os.path.join(param_base_dir, "cst_ckpt")  # 可以断点训练的模型保存路径

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

star_program = fluid.default_startup_program()

# 定义数据读取器
batch_size = 2


def get_batches(Xs, ys, train_len):
    def reader():
        for i in range(train_len):
            train_xs = Xs[i]
            train_ys = ys[i]
            yield train_xs, train_ys

    return reader


batch_num = (len(train_xs) // batch_size)

main_program = fluid.default_main_program()
avg_cost, opt, accuracy = train_program()

feed_order = ['input_tensor', 'labels']

feed_var_list_loop = [
    main_program.global_block().var(var_name) for var_name in feed_order
]
feeder = fluid.DataFeeder(
    feed_list=feed_var_list_loop, place=place)

# test_program = main_program.clone(for_test=True)  # 测试计算图和训练用的一样，for_test参数会为测试用途进行优化，加快执行速度

# 参数初始化
exe.run(star_program)


# fluid.io.load_persistables(executor=exe, dirname=ckpt_param_path,main_program=main_program)

# 网络训练函数
def training(data_train, step_id, epoch_i, batch_i, batch_num):
    cost, acc = exe.run(main_program,
                        feed=feeder.feed(data_train),
                        fetch_list=[avg_cost, accuracy])
    if (step_id % 20 == 0):
        time_str = datetime.datetime.now().isoformat()
        print('Training {}: Epoch {:>3}  Batch {:>4} /{}   loss {:.5f}  accuracy {:.5f}'.format(
            time_str, epoch_i, batch_i, batch_num, cost[0], acc[0]))
        print('cost', cost)
    return acc, cost


# # 网络测试函数
# def testing(data_test,step_id,epoch_i,batch_i,batch_num):
#     cost, acc = exe.run(test_program,
#                             feed=feeder.feed(data_test),
#                             fetch_list=[avg_cost,accuracy])
#     if (step_id % 20 == 0):
#         time_str = datetime.datetime.now().isoformat()
#         print('Testing {}: Epoch {:>3}  Batch {:>4} /{}   loss {:.5f}  accuracy {:.5f}'.format(
#             time_str,epoch_i,batch_i,batch_num,cost[0],acc[0]))
#     return acc,cost


# 开始训练
epoches = 30

best_loss = 9999

for ii in range(epoches):
    train_reader = paddle.batch(get_batches(train_xs, train_ys, len(train_xs)), batch_size)
    for batch_id, data_train in enumerate(train_reader()):
        training(data_train, ii * batch_num + batch_id, ii, batch_id, batch_num)

# 保存模型
fluid.io.save_persistables(exe, ckpt_param_path, main_program)

# # 计算验证集的循环次数
# batch_num = (len(valid_xs) // batch_size)
# test_loss = 0.0
# test_acc = 0.0
# # 遍历所有验证集数据
# valid_reader = paddle.batch(get_batches(valid_xs,valid_ys,len(valid_xs)),batch_size)
# for batch_id, data_valid in enumerate(valid_reader()):
#     acc, loss = testing(data_valid, ii * batch_num + batch_id, ii, batch_id, batch_num)
#     test_loss = test_loss + loss
#     test_acc = test_acc + acc

# test_acc = test_acc / batch_num
# test_loss = test_loss / batch_num
# if test_loss < best_loss:
#     best_loss = test_loss
#     print("best loss = {}  acc = {}".format(best_loss, test_acc))
# else:
#     print("test loss = {}  acc = {}".format(test_loss, test_acc))

