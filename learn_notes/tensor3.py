#多元线性回归
import tensorflow as  tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv('11111.csv')
print(df.describe())
#print(df)
df = df.values
#装换数组
df = np.array(df)

#每一列都要归一话 避免过大过小的影响 权重影响太大
for i in range(6):
    df[:,i] = df[:,i]/(df[:,i].max()-df[:,i].min())

x_data = df[:, :4]#前4列
y_data = df[:, 5]#第5列

print(x_data)
print(y_data)


#12个特征数据12列,要归一话 处理  1个标签数据1列
x = tf.placeholder(tf.float32,[None,4],name = "X")
y = tf.placeholder(tf.float32,[None,1],name = "Y")

"""y1 = tf.placeholder(tf.float32,[None,7],name = "Y1")

y2 = tf.placeholder(tf.float32,[None,4],name = "Y2")"""

#打包
with tf.name_scope("Model"):
    #w初始化值微shape=(12,1)的随机数
    w = tf.Variable(tf.random_normal([4,1],stddev=0.01),name = "W")
    b = tf.Variable(tf.zeros([1]))
    pred = tf.matmul(x,w)+b
"""    y1 = tf.matmul(x, w) + b
    y1 =tf.nn.relu(y1)

    w1 = tf.Variable(tf.random_normal([7,4],stddev=0.01),name = "W1")
    b1 = tf.Variable(tf.zeros([4]))

    y2 = tf.matmul(y1, w1) + b1
    y2 =tf.nn.relu(y2)

    w2 = tf.Variable(tf.random_normal([4,1],stddev=0.01),name = "W2")
    b2 = tf.Variable(tf.zeros([1]))"""


train_epochs = 500
learning_rate = 0.01

with tf.name_scope("LossFunction"):
    loss_function = tf.reduce_mean(tf.pow(y-pred,2))
    optimizer =  tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

sess = tf.Session()
init =tf.global_variables_initializer()

sess.run(init)

loss_list = []
for epoch in range (train_epochs):
    loss_sum =0.0
    for xs,ys in zip(x_data,y_data):
        #数组转换向量
        xs = xs.reshape(1,4)
        ys = ys.reshape(1,1)
        _,loss = sess.run([optimizer,loss_function],feed_dict={x:xs,y:ys})
        loss_sum = loss_sum +loss

    ##打乱数据对 顺序
    xvalues,yvalues = shuffle(x_data,y_data)
    b0temp = b.eval(session = sess)
    w0temp = w.eval(session = sess)
    loss_average = loss_sum/len(y_data)
    loss_list.append(loss_average)
    if epoch%5 == 0:
        print("epoch=",epoch+1,"loss=",loss_average,"b=",b0temp)
        print("w=",w0temp)

plt.plot(loss_list)
plt.show()


##测试
n = np.random.randint(14)
#print(n)
x_test = x_data[n]##随机挑选数据
print(x_test)

x_test =x_test.reshape(1,4)
predict = sess.run(pred,feed_dict = {x:x_test})
print("预测值:%f" %predict)
target = y_data[n]
print("标签值:%f"%target)
sess.close()


