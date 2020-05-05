import os
import numpy as np
import pickle
import paddle.fluid as fluid
import paddle
import seaborn as sns
import matplotlib.pyplot as plt
import math
# 加载数据集


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


data_dir = './data/cp.txt'
text = load_data(data_dir)


# print(text)


# 数据预处理

# 单词到ID的转换字典： vocab_to_int
# ID到单词的转换字典： int_to_vocab


def create_lookup_tables():
    vocab_to_int_1 = {str(ii).zfill(3) : ii for ii in range(1000)}
    int_to_vocab_1 = {ii: str(ii).zfill(3) for ii in range(1000)}
    return vocab_to_int_1, int_to_vocab_1


# 将每期结果按照从第一期开始的顺序保存到文件中。


text = load_data(data_dir)

words = [word for word in text.split()]

reverse_words = [text.split()[idx] for idx in (range(len(words)-1, 0, -1))]
vocab_to_int, int_to_vocab = create_lookup_tables()

# 将升序排列的所有号码转成对应的索引ID,作为输入数据
int_text = [vocab_to_int[word] for word in reverse_words]
# 将预处理后的数据保存到本地
pickle.dump((int_text, vocab_to_int, int_to_vocab), open('preprocess.p', 'wb'))


# 读取保存的数据
int_text, vocab_to_int, int_to_vocab = pickle.load(open('preprocess.p', mode='rb'))

# 定义函数用来取得批量数据
def get_batches(int_text_1, batch_size_1, seq_length_1):
    def reader():
        batchcount_1 = len(int_text_1) // (batch_size_1 * seq_length_1)
        int_text_inputs = int_text_1[:batchcount_1 * (batch_size_1 * seq_length_1)]
        int_text_targets = int_text_1[1:batchcount_1 * (batch_size_1 * seq_length_1)+1]
        for ii in range(len(int_text_inputs)):
            yield [int_text_inputs[ii]], [int_text_targets[ii]]
    return reader


'''
超参数
'''
# 训练迭代次数
epochs = 3 # 批次大小
batch_size = 32
# RNN的大小（隐藏节点的维度）
rnn_size = 512
# 嵌入层的维度
embed_dim = 512
# 序列的长度，始终为1
seq_length = 1
# 学习率
learning_rate = 0.01
# 过多少batch以后打印训练信息
show_every_n_batches = 10

save_dir = './save'

# 构建计算图


vocab_size = len(int_to_vocab)
# 定义输入、目标和学习率占位符
input_text = fluid.data('input', shape=[None, 1], dtype='int64', lod_level=1)
targets = fluid.data('targets', shape=[None, 1], dtype='int64')


# def test():
#     ###############################################
#
#     # 优化器
#     def optimizer(cross_entropy):
#         # 梯度裁剪
#         fluid.clip.set_gradient_clip(
#             fluid.clip.GradientClipByValue(min=-1.0, max=1.0))
#         optimizer = fluid.optimizer.AdamOptimizer(0.01)
#         train_step = optimizer.minimize(cross_entropy)
#         return train_step
#
#     ##################################################
#     ###################
#
#     embed_layer = fluid.layers.embedding(input=input_text, size=[vocab_size, embed_dim])
#     print(input_text.lod_level)
#     print('embed_layer shape', embed_layer.shape)
#
#     print('input_text shape', input_text.shape)
#     # drnn = fluid.layers.DynamicRNN()
#     # with drnn.block():
#     #     # 定义单步输入
#     #     word = drnn.step_input(embed_layer)
#     #     print('word shape', word.shape)
#     #     # 定义第一层lstm的hidden_state, cell_state
#     #     prev_hid0 = drnn.memory(shape=[rnn_size])
#     #     prev_cell0 = drnn.memory(shape=[rnn_size])
#     #
#     #     # # 定义第二层lstm的hidden_state, cell_state
#     #     # prev_hid1 = drnn.memory(shape=[rnn_size])
#     #     # prev_cell1 = drnn.memory(shape=[rnn_size])
#     #
#     #
#     #     print('prev_cell1 ', prev_cell0)
#     #
#     #
#     #     # 执行两层lstm运算
#     #     cur_hid0, cur_cell0 = fluid.layers.lstm_unit(word, prev_hid0, prev_cell0)
#     #     # cur_hid1, cur_cell1 = fluid.layers.lstm_unit(cur_hid0, prev_hid1, prev_cell1)
#     #
#     #     # 更新第一层lstm的hidden_state, cell_state
#     #     drnn.update_memory(prev_hid0, cur_hid0)
#     #     drnn.update_memory(prev_cell0, cur_cell0)
#     #
#     #     # 更新第二层lstm的hidden_state, cell_state
#     #     # drnn.update_memory(prev_hid1, cur_hid1)
#     #     # drnn.update_memory(prev_cell1, cur_cell1)
#     #
#     #     drnn.output(cur_hid0)
#     #
#     # outputs = drnn()
#     # print('outputs ', outputs)
#     # last = fluid.layers.sequence_last_step(outputs)
#
#     forward_proj = fluid.layers.fc(input=embed_layer, size=rnn_size * 4, bias_attr=False)
#     forward, cell = fluid.layers.dynamic_lstm(input=forward_proj, size=rnn_size * 4, use_peepholes=False)
#     logits = fluid.layers.fc(forward, size=vocab_size)
#     # print('logits shape ', logits.shape)
#     # probs = fluid.softmax(logits, name='probs')
#
#     # soft_label 参数默认为 False ,即非  one_hot 模式
#     cost = fluid.layers.softmax_with_cross_entropy(logits, targets)
#
#     # print('cost shape ', cost.shape)
#     cross_entropy = fluid.layers.reduce_mean(cost)
#     optimizer(cross_entropy)
#     ############################################################################
#
#     #################################################
#     use_cuda = False
#     place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
#     exe = fluid.Executor(place)
#
#     ####################################
#
#     ###########################
#     feed_order = ['input', 'targets']
#
#     feed_var_list_loop = [
#         fluid.default_main_program().global_block().var(var_name) for var_name in feed_order
#     ]
#
#     feeder = fluid.DataFeeder(
#         feed_list=feed_var_list_loop, place=place)
#
#     exe.run(fluid.default_startup_program())
#
#     ##########################################
#
#     ############################
#     train_reader = paddle.batch(get_batches(int_text[:-(batch_size + 1)], batch_size, seq_length), batch_size)
#     for batch_i, data_train in enumerate(train_reader()):
#         results = exe.run(fluid.default_main_program(),
#                         feed=feeder.feed(data_train),
#                         fetch_list=[cross_entropy], return_numpy=False)
#     # print('results[0] ', np.array(results[0]))
#     # print('results shape', np.array(results[0]).shape)
#
#
#     #########################
# test()

input_data_shape = input_text.shape

# 嵌入矩阵
embed_matrix_np = np.random.uniform(size=(vocab_size, embed_dim), low=-1, high=1)
w_param_attrs = fluid.ParamAttr(
    name="emb_weight",
    initializer=fluid.initializer.NumpyArrayInitializer(embed_matrix_np),
    trainable=True)
embed_layer = fluid.layers.embedding(input=input_text, size=[vocab_size, embed_dim], param_attr=w_param_attrs)

# 使用余弦距离计算相似度
# embed_matrix_tmp = embed_matrix_np.astype("float32")
# embed_matrix = fluid.layers.assign(embed_matrix_tmp)
norm = np.sqrt(np.sum(np.square(embed_matrix_np), 1, keepdims=True))
normalized_embedding = embed_matrix_np / norm
# 构建RNN单元
def network(input_text):
    forward_proj = fluid.layers.fc(input=embed_layer, size=rnn_size * 4, bias_attr=False)
    forward, cell = fluid.layers.dynamic_lstm(input=forward_proj, size=rnn_size * 4, use_peepholes=False)
    logits = fluid.layers.fc(forward, size=vocab_size)

    # soft_label 参数默认为 False ,即非  one_hot 模式
    cost = fluid.layers.softmax_with_cross_entropy(logits, targets)
    cross_entropy = fluid.layers.reduce_mean(cost)
    # print('cross_entropy  ', cross_entropy)
    # 对网络的输出做softmax, 得到1000个号码的概率向量
    # print('logits shape', logits.shape)
    probs = fluid.layers.softmax(logits)
    
    # print('probs  ', probs)

    # 准确率
    correct_pred = fluid.layers.argmax(probs, 1) == fluid.layers.argmax(fluid.layers.cast(targets, dtype='int64'), 1)#logits <--> probs  tf.argmax(targets, 1) <--> targets
    accuracy = fluid.layers.reduce_mean(fluid.layers.cast(correct_pred, dtype='float32'))
    # print('accuracy  ', accuracy)

    return cross_entropy, accuracy, probs


#     # 使用RNN单元构建RNN

# 准备批量数据
batches = get_batches(int_text[:-(batch_size + 1)], batch_size, seq_length)
test_batches = get_batches(int_text[-(batch_size + 1):], batch_size, seq_length)
top_k = 10
# 训练数据和测试数据的batchnum
batch_cnt_train = (len(int_text[:-(batch_size + 1)]) // (batch_size * seq_length))
batch_cnt_test = (len(int_text[-(batch_size + 1):]) // (batch_size * seq_length))

# 预测结果的Top K 准确率
topk_acc_list = []
topk_acc = 0
# 与预测结果距离最近的Top K 准确率
sim_topk_acc_list = []
sim_topk_acc = 0
# 表示 K 值是一个范围，不想Top K是最开始的K个
range_k = 5
# 以每次训练得出的距离中位数为中心，以范围K为半径的准确率，使用预测结果向量
floating_median_idx = 0
floating_median_acc_range_k = 0
floating_median_acc_range_k_list = []
# 同上，使用的是相似度向量
floating_median_sim_idx = 0
floating_median_sim_acc_range_k = 0
floating_median_sim_acc_range_k_list = []
# 保存训练损失和测试损失
losses = {'train':[], 'test':[]}


acc_list = []
# 保存各类准确率
accuracies = {'accuracy':[], 'topk':[], 'sim_topk':[], 'floating_median_acc_range_k':[], 'floating_median_sim_acc_range_k':[]}

# 优化器
def optimizer(cross_entropy):
    # 梯度裁剪
    fluid.clip.set_gradient_clip(
        fluid.clip.GradientClipByValue(min=-1.0, max=1.0))
    optimizer = fluid.optimizer.AdamOptimizer(0.01)
    train_step = optimizer.minimize(cross_entropy)
    return train_step


cost, accuracy, probs = network(input_text)
train_step = optimizer(cost)


# batchCnt = len(int_text) // (batch_size * seq_length)
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
main_program = fluid.default_main_program()
test_program = main_program.clone(for_test=True)  # 测试计算图和训练用的一样，for_test参数会为测试用途进行优化，加快执行速度


feed_order = ['input', 'targets']

feed_var_list_loop = [
    main_program.global_block().var(var_name) for var_name in feed_order
]

feeder = fluid.DataFeeder(
    feed_list=feed_var_list_loop, place=place)


def training(data_train, epoch_i, batch_i, batchCnt):
    train_loss = exe.run(main_program,
                        feed=feeder.feed(data_train),
                        fetch_list=[cost])

    # Show every <show_every_n_batches> batches
    # 打印训练信息
    losses['train'].append(np.array(train_loss)[0])
    if (epoch_i * batchCnt + batch_i) % show_every_n_batches == 0:
        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
            epoch_i,
            batch_i,
            batch_cnt_train,
            train_loss[0][0]))



# 测试函数
def testing(data_train, epoch_i, batch_i, batchCnt):
    test_loss, acc, probs1, lay = exe.run(test_program,
                        feed=feeder.feed(data_train),
                        fetch_list=[cost, accuracy, probs, embed_layer], return_numpy=False)

    # 保存测试损失和准确率
    acc_list.append(np.array(acc)[0])
    losses['test'].append(np.array(test_loss)[0])
    accuracies['accuracy'].append(np.array(acc)[0])
    # Show every <show_every_n_batches> batches
    # 打印训练信息
    print('test Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
            epoch_i,
            batch_i,
            batch_cnt_test,
            np.array(test_loss)[0]))
    return probs1


# 训练网络
exe.run(fluid.default_startup_program())
for epoch_i in range(epochs):
    train_reader = paddle.batch(batches, batch_size)
    for batch_i, data_train in enumerate(train_reader()):
        training(data_train, epoch_i, batch_i, batch_cnt_train)


    # 测试网络
    train_reader = paddle.batch(test_batches, batch_size)
    for batch_i, data_train in enumerate(train_reader()):
        probs2 = testing(data_train, epoch_i, batch_i, batch_cnt_test)

        # valid_embedding = tf.nn.embedding_lookup(normalized_embedding, np.squeeze(probabilities.argmax(2)))
        main_program_1 = fluid.Program()
        # 如果您不需要关心startup program,传入一个临时值即可
        with fluid.program_guard(main_program_1, fluid.Program()):
            # # 利用嵌入矩阵和生成的预测计算得到相似度矩阵sim
            w_param_attrs_norm = fluid.ParamAttr(
                name="emb_weight",
                initializer=fluid.initializer.NumpyArrayInitializer(normalized_embedding),
                trainable=True)
            probs2_tensor = fluid.layers.assign(np.array(probs2))
            arg = fluid.layers.argmax(x=probs2_tensor, axis=1)
            arg1 = fluid.layers.unsqueeze(arg, 1)
            valid_embedding = fluid.layers.embedding(input=arg1,
                                                     size=[vocab_size, embed_dim], param_attr=w_param_attrs_norm)
            normalized_embedding_tmp = normalized_embedding.astype("float32")
            normalized_embedding_ts = fluid.layers.assign(normalized_embedding_tmp)
            sim2 = fluid.layers.matmul(valid_embedding, fluid.layers.transpose(normalized_embedding_ts, [1, 0]))
            sim = exe.run(main_program_1, fetch_list=[sim2])


        # 保存预测结果的Top K准确率和与预测结果距离最近的Top K准确率
        topk_acc = 0
        sim_topk_acc = 0
        for ii in range(len(np.array(probs2))):
            sim1 = -sim[0][ii, :]
            nearest = sim1.argsort()[0:top_k]
            if data_train[ii][1] in nearest:
                sim_topk_acc += 1

            if data_train[ii][1] in (-np.array(probs2)[ii]).argsort()[0:top_k]:
                topk_acc += 1

        topk_acc = topk_acc / len(data_train)
        topk_acc_list.append(topk_acc)
        accuracies['topk'].append(topk_acc)

        sim_topk_acc = sim_topk_acc / len(data_train)
        sim_topk_acc_list.append(sim_topk_acc)
        accuracies['sim_topk'].append(sim_topk_acc)

        # 计算真实值在预测值中的距离数据
        realInSim_distance_list = []
        realInPredict_distance_list = []
        for ii in range(len(np.array(probs2))):
            sim_nearest = (-sim[0][ii, :]).argsort()
            idx = list(sim_nearest).index(data_train[ii][1])
            realInSim_distance_list.append(idx)

            nearest = (-np.array(probs2)[ii]).argsort()
            idx = list(nearest).index(data_train[ii][1])
            realInPredict_distance_list.append(idx)
        print('真实值在预测值中的距离数据：')
        print('max distance : {} '.format(max(realInPredict_distance_list)))
        print('min distance : {}'.format(min(realInPredict_distance_list)))
        print('平均距离 : {}'.format(np.mean(realInPredict_distance_list)))
        print('距离中位数 : {}'.format(np.median(realInPredict_distance_list)))
        print('距离标准差 : {}'.format(np.std(realInPredict_distance_list)))

        print('真实值在预测值相似向量中的距离数据：')
        print('max distance : {}'.format(max(realInSim_distance_list)))
        print('min distance : {}'.format(min(realInSim_distance_list)))
        print('平均距离 : {}'.format(np.mean(realInSim_distance_list)))
        print('距离中位数 : {}'.format(np.median(realInSim_distance_list)))
        print('距离标准差 : {}'.format(np.std(realInSim_distance_list)))

        # 计算以距离中位数为中心，范围K为半径的准确率
        floating_median_sim_idx = int(np.median(realInSim_distance_list))
        floating_median_sim_acc_range_k = 0

        floating_median_idx = int(np.median(realInPredict_distance_list))
        floating_median_acc_range_k = 0
        for ii in range(len(np.array(probs2))):
            nearest_floating_median = (-np.array(probs2)[ii]).argsort()[
                                      floating_median_idx - range_k:floating_median_idx + range_k]
            if data_train[ii][1] in nearest_floating_median:
                floating_median_acc_range_k += 1

            nearest_floating_median_sim = (-sim[0][ii, :]).argsort()[
                                          floating_median_sim_idx - range_k:floating_median_sim_idx + range_k]
            if data_train[ii][1] in nearest_floating_median_sim:
                floating_median_sim_acc_range_k += 1

        floating_median_acc_range_k = floating_median_acc_range_k / len(data_train)
        floating_median_acc_range_k_list.append(floating_median_acc_range_k)
        accuracies['floating_median_acc_range_k'].append(floating_median_acc_range_k)

        floating_median_sim_acc_range_k = floating_median_sim_acc_range_k / len(data_train)
        floating_median_sim_acc_range_k_list.append(floating_median_sim_acc_range_k)
        accuracies['floating_median_sim_acc_range_k'].append(floating_median_sim_acc_range_k)

    print('Epoch {:>3} floating median sim range k accuracy {}'.format(
        epoch_i, np.mean(floating_median_sim_acc_range_k_list)))  # :.3f
    print('Epoch {:>3} floating median range k accuracy {} '.format(
        epoch_i, np.mean(floating_median_acc_range_k_list)))  # :.3f
    print('Epoch {:>3} similar top k accuracy {} '.format(epoch_i, np.mean(sim_topk_acc_list)))  # :.3f
    print('Epoch {:>3} top k accuracy {} '.format(epoch_i, np.mean(topk_acc_list)))  # :.3f
    print('#############  acclist ', acc_list)
    print('Epoch {:>3} accuracy {} '.format(epoch_i, np.mean(acc_list)))   # :.3f

sns.distplot(realInSim_distance_list, rug=True)
plt.show()
sns.distplot(realInPredict_distance_list, rug=True)
plt.show()
plt.plot(losses['train'], label='Training loss')
plt.legend()
_ = plt.ylim()
plt.show()
plt.plot(losses['test'], label='Test loss')
plt.legend()
_ = plt.ylim()
plt.show()

plt.plot(accuracies['accuracy'], label='Accuracy')
plt.plot(accuracies['topk'], label='Top K')
plt.plot(accuracies['sim_topk'], label='Similar Top K')
plt.plot(accuracies['floating_median_acc_range_k'], label='Floating Median Range K Acc')
plt.plot(accuracies['floating_median_sim_acc_range_k'], label='Floating Median Sim Range K Acc')
plt.legend()
_ = plt.ylim()
plt.show()


test_reader = paddle.batch(test_batches, batch_size)
for batch_i, data_test in enumerate(test_reader()):
    # (data_test_x, data_test_y) = data_test
    data_test_y = [data_train[ii][1] for ii in range(len(data_train))]
    plt.plot(data_test_y, label='Targets')
    plt.plot(np.array(probs2).argmax(1), label='Prediction')
    plt.legend()
    _ = plt.ylim()
plt.show()
