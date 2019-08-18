#！/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Kyle Chen time:

import os
import tensorflow as tf
import numpy as np
import time
import xlwt

# 数据导入(将csv文件转换为tfrecords文件导入)
dir1 = 'E:/TestAndTrain/train_data/'
dir2 = 'E:/TestAndTrain/test_data/'
classes = {'driverA', 'driverB', 'driverC'}  # 人为设定3类
writer = tf.python_io.TFRecordWriter("driver_train.tfrecords")  # 要生成的文件

# 搭建RNN网络
# RNN各种参数定义
lr = tf.Variable(0.01, dtype=tf.float32)  # 学习速率
training_iters = 25  # 循环次数
batch_size =1 # 每一批次大小设定为1
n_inputs = 5  # 每一个csv文件的大小是8743*5，这里是手写字中的每行5列的数值
n_steps = 8743  # 这里是每个文件中8743行的数据，因为以一行一行值处理的话，正好是8743行
n_hidden_units = 25  # 假设隐藏单元有40个
n_classes = 3  # 因为驾驶员有3个，因此最后要分成3个类
d = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
e = {0: '驾驶员 A', 1: '驾驶员 B', 2: '驾驶员 C'}
# 定义输入和输出的placeholder
x = tf.placeholder(tf.float32, [n_steps, n_inputs])
y = tf.placeholder(tf.float32, [1, n_classes])

# 对weights和biases初始值定义
weights = {
    # shape(5, 30)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape(40 , 3)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    # shape(30, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape(3, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X,  weights,  biases):
    X_in = tf.matmul(X, weights['in']) + biases['in']  # Xin shape 8743,40
    X_in = tf.reshape(X_in, [1, n_steps, n_hidden_units])  #Xin shape 1,8743,40
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    results = tf.matmul(states[1], weights['out']) + biases['out']
    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 训练部分
    step = 0
    start = time.clock()
    class_path = dir2 + '/'
    excel = xlwt.Workbook()
    sheet1 = excel.add_sheet('预测结果')
    sheet2 = excel.add_sheet('运算消耗时间')
    sheet1.write(0, 0, '预测对象')
    sheet1.write(0, 1, '预测结果')
    sheet2.write(0, 0, '运算对象')
    sheet2.write(0, 1, '运算消耗时间/s')
    sheet2.write(1, 0, '训练部分')
    for index, name in enumerate(classes):
        sess.run(tf.assign(lr, 0.01*(0.65**step)))
        step = step + 1
        class_path1 = dir1 + name + '/'
        for file_name in os.listdir(class_path1):
            file_path = class_path1 + file_name  # 每一个csv文件的地址
            file = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=0)
            file = file[0:8743]
            label = np.array(d[index]).reshape([1, 3])
            sess.run([train_op], feed_dict={x: file, y: label})
    end = time.clock()
    print('训练阶段的时间花费为 ' + str(end-start))
    sheet2.write(1, 1, str(end-start))

    # 测试部分
    m = 2
    for file_name in os.listdir(dir2):
        start = time.clock()
        file_path = dir2 + file_name  # 每一个csv文件的地址
        file = np.loadtxt(open(file_path, "rb"), delimiter=",", skiprows=1)
        file = file[0:8743]
        file = file[:, 1:6]
        predic = sess.run(pred, feed_dict={x: file})
        max_pos = 0
        for i in range(3):
            tmp = predic[0][max_pos]
            if predic[0][i] > tmp:
                max_pos = i
                i += 1
        print('数据 ' + file_name + ' 的预测结果为 ' + e[max_pos])
        end = time.clock()
        print('针对数据' + file_name + '的测试阶段的时间花费为 ' + str(end-start))
        # 将测试得到的数据写入Excel
        sheet2.write(m, 0, file_name)
        sheet2.write(m, 1, str(end-start))
        sheet1.write(m-1, 0, file_name)
        sheet1.write(m-1, 1, e[max_pos])
        m = m+1
        excel.save('E:/test_result.xlsx')


