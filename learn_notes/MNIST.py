from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

'''print('训练集 数量:',mnist.train.num_examples,
      ',验证集数量:',mnist.validation.num_examples,
      ',测试集数量:',mnist.test.num_examples)
print('train images shape:',mnist.train.images.shape,
      'labels shape:',mnist.train.labels.shape)
plt.imshow(mnist.train.images[12].reshape(14,56),cmap = 'binary')
print(mnist.train.labels[0:10])
plt.show()'''

x = tf.placeholder(tf.float32,[None,784],name="X")
y = tf.placeholder(tf.float32,[None,10],name="Y")

W = tf.Variable(tf.random_normal([784,10]),name="W")
b = tf.Variable(tf.zeros([10]),name="b")

forward = tf.matmul(x,W)+b
pred = tf.nn.softmax(forward)#jihuo

train_epoch = 50  #训练轮数
batch_size = 100  #单次训练批次大小
total_batch = int(mnist.train.num_examples/batch_size)# 一轮训练的批次
display_step = 1
learning_rate =0.01


loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))#返回真挚 匹配情况
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#浮点数 准确率并且计算平均值

sess = tf.Session()
init =tf.global_variables_initializer()
sess.run(init)

for epoch in range(train_epoch):
      for batch in range(total_batch):
            xs,ys = mnist.train.next_batch(batch_size)# next_batch下一个组批次 打乱
            sess.run(optimizer,feed_dict={x:xs, y:ys})

      loss,acc = sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images,y:mnist.validation.labels})

      if(epoch+1)%display_step ==0:
            print("Train Epoch:",'%02d' % (epoch+1),"Loss=","{:.9f}".format(loss),"Accurary=","{:.4f}".format(acc))

print("train finshed")
sess.close()


