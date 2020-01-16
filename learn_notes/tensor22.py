#多元线性回归

import tensorflow as  tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import keras
from keras import layers
from keras.utils.visualize_util import plot
import h5py

df = pd.read_csv('hhh.csv')
print(df.describe())
#print(df)
df = df.values
#装换数组
df = np.array(df)

#每一列都要归一话 避免过大过小的影响 权重影响太大
for i in range(6):
    df[:,i] = df[:,i]/(df[:,i].max()-df[:,i].min())

x_data = df[:, :5]#前4列
y_data = df[:, 5]#第5列

print(x_data)
print(y_data)


#12个特征数据12列,要归一话 处理  1个标签数据1列
x = tf.placeholder(tf.float32,[None,5],name = "X")
y = tf.placeholder(tf.float32,[None,1],name = "Y")

"""y1 = tf.placeholder(tf.float32,[None,7],name = "Y1")

y2 = tf.placeholder(tf.float32,[None,4],name = "Y2")"""

# #打包
# with tf.name_scope("Model"):
#     #w初始化值微shape=(12,1)的随机数
#     w = tf.Variable(tf.random_normal([4,1],stddev=0.01),name = "W")
#     b = tf.Variable(tf.zeros([1]))
#     pred = tf.matmul(x,w)+b
# """    y1 = tf.matmul(x, w) + b
#     y1 =tf.nn.relu(y1)
#
#     w1 = tf.Variable(tf.random_normal([7,4],stddev=0.01),name = "W1")
#     b1 = tf.Variable(tf.zeros([4]))
#
#     y2 = tf.matmul(y1, w1) + b1
#     y2 =tf.nn.relu(y2)
#
#     w2 = tf.Variable(tf.random_normal([4,1],stddev=0.01),name = "W2")
#     b2 = tf.Variable(tf.zeros([1]))"""


model = keras.models.Sequential()
model.add(layers.Dense(15, activation='tanh', input_dim=5))
model.add(layers.Dense(15, activation='tanh'))
model.add(layers.Dense(15, activation='tanh'))
model.add(layers.Dense(1, activation='tanh'))
train_epochs = 800
model.compile(optimizer='Adagrad', loss='mse')


# x_data输入网络中，得到预测值y_pred


# with tf.name_scope("LossFunction"):
#     loss_function = tf.reduce_mean(tf.pow(y-pred,2))
#     optimizer =  tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = tf.Session()
init =tf.global_variables_initializer()

sess.run(init)
# 训练3001个批次
for step in range(3001):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data, y_data)
    if step % 100 == 0:
        print('cost:', cost)

for i in  range(3):
    w, b = model.layers[0].get_weights()
    print('w:', w, 'b:', b)

# 打印权值和偏置值


# loss_list = []
# for epoch in range (train_epochs):
#     loss_sum =0.0
    # for xs,ys in zip(x_data,y_data):
    #     #数组转换向量
    #     xs = xs.reshape(1,4)
    #     ys = ys.reshape(1,1)
    #     _,loss = sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
    #     loss_sum = loss_sum +loss

    ##打乱数据对 顺序
    # xvalues,yvalues = shuffle(x_data,y_data)
    # b0temp = b.eval(session = sess)
    # w0temp = w.eval(session = sess)
    # loss_average = loss_sum/len(y_data)
    # loss_list.append(loss_average)
    # if epoch%5 == 0:
    #     print("epoch=",epoch+1,"loss=",loss_average)
    #     print("w=",w0temp)

# plt.plot(loss_list)
# plt.show()
if cost<0.223:
    model.save_weights('./weights.h5')


#测试
for i in range(18):
    # print(n)
    x_test = x_data[i]  ##随机挑选数据
    print(x_test)

    x_test = x_test.reshape(1, 5)
    # predict = sess.run(pred,feed_dict = {x:x_test})
    y_pred = model.predict(x_test)
    print("预测值:%f" % y_pred)
    target = y_data[i]
    print("标签值:%f" % target)

hist = model.fit(x_test, y_test, batch_size=10, nb_epoch=100, shuffle=True,verbose=0,validation_split=0.2)
plot(hist)
sess.close()



