##单变量线性回归
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#设置随机数种子 不同的数值代表不同的随机数 保证实验一致性,保证随机数一样
np.random.seed(5)

#100个等差随机数
x_data = np.linspace(-1,1,200)
#y=2x+1 加入噪声维度和数组一样
y_data = 2*x_data+1.0+np.random.randn(*x_data.shape) * 0.4

plt.subplot(221)
plt.scatter(x_data,y_data)
plt.plot(x_data,1+2*x_data,color ='red',linewidth = 3)




x = tf.placeholder("float",name = "x")
y = tf.placeholder("float",name = "y")
#模型
def model(x,w,b):
    return tf.multiply(x,w)+b

w = tf.Variable(1.0,name = "w0")

b = tf.Variable(0.0, name ="b0" )
pred = model(x,w,b)

#训练模型
#迭代次数
train_epochs = 30
#学习率
learning_rate = 0.05
#损失函数 均方差
loss_function = tf.reduce_mean(tf.square(y-pred))
#梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

###会话

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
step = 0 #记录训练的步数
loss_list = [] #保存loss的值


#开始训练 轮数为epoch 采用SGD随机梯度下降优化
for epoch in range(train_epochs):
    for xs,ys in zip(x_data,y_data):  #SGD
        _, loss=sess.run([optimizer,loss_function],feed_dict ={x:xs,y:ys})

        #按照步长显示loss
        loss_list.append(loss)
        step=step+1
        display_step =10
        if step % display_step ==0:
                print("Train Epoch:",'%02d' % (epoch+1),"Step: %03d" % (step),"loss=","{:.9f}".format(loss))

    b0temp=b.eval(session=sess)
    w0temp=w.eval(session=sess)
    plt.subplot(222)
    plt.plot(x_data,w0temp * x_data + b0temp)  #画图



#可视化
plt.scatter(x_data,y_data,label='Original data')
plt.plot(x_data,x_data*sess.run(w)+sess.run(b),label='Fitted line',color = 'r',linewidth = 3)
plt.legend(loc = 10)#通过参数loc指定图例的位置
plt.figure(2)
plt.plot(loss_list,'r^')
plt.show()


print('w:',sess.run(w))
print('b:',sess.run(b))


###预测1
x_test=3.21
predict = sess.run(pred,feed_dict={x:x_test})
print("预测值:%f" % predict)
target = 2*x_test + 1.0
print("目标值:%f" % target)

###预测2有问题
x_test1=3.21
predict = sess.run(w)*x_test1 + sess.run(b)
print("预测值:%f" % predict)
target = 2*x_test + 1.0
print("目标值:%f" % target)
sess.close()