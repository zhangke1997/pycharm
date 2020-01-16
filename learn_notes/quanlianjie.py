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

#隐藏曾
H1_NN =256
W1 = tf.Variable(tf.random_normal([784,H1_NN]))
b1 = tf.Variable(tf.zeros([H1_NN]))

Y1 = tf.nn.relu(tf.matmul(x,W1)+b1)#jihuo

#输出层
W2 = tf.Variable(tf.random_normal([H1_NN,10]))
b2 = tf.Variable(tf.zeros([10]))
forward = tf.matmul(Y1,W2)+b2
pred = tf.nn.softmax(forward)

train_epoch = 40  #训练轮数
batch_size = 50  #单次训练批次大小
total_batch = int(mnist.train.num_examples/batch_size)# 一轮训练的批次
display_step = 1
learning_rate =0.01

#交叉商
#log(0)值为NaN造成数据不稳定
#loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= forward,labels= y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))#返回真挚 匹配情况
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#浮点数 准确率并且计算平均值

from time import time
startTime = time()

sess = tf.Session()
init =tf.global_variables_initializer()
sess.run(init)

for epoch in range(train_epoch):
      for batch in range(total_batch):
            xs,ys = mnist.train.next_batch(batch_size)# next_batch下一个组批次 打乱
            sess.run(optimizer,feed_dict={x: xs, y: ys})

      loss,acc = sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images,y:mnist.validation.labels})

      if(epoch+1)%display_step ==0:
            print("Train Epoch:",'%02d' % (epoch+1),"Loss=","{:.9f}".format(loss),"Accurary=","{:.4f}".format(acc))

print("train finshed")
duration = time()-startTime
print("Train Finished takes:","{:.2f}".format((duration)))


#评估
accu_test = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
print("Test Accuracy:",accu_test)


#预测
#pred 是one_hot编码格式,argmax转换0到9
prediction_result =sess.run(tf.argmax(pred,1),
                        feed_dict = {x:mnist.test.images})
print(prediction_result[0:10])

#找出预测错误
def print_predict_errs(labels,prediction):
      count = 0
      compare_lists = (prediction == np.argmax(labels,1))
      err_lists = [i for i in range(len(compare_lists)) if compare_lists[i] == False]
      for x in err_lists:
            print("index="+str(x)+
                  "标签值=", np.argmax(labels[x]),
                  "预测值=", prediction[x])
            count = count+1
      print("总结:"+str(count))

#调用
print_predict_errs(labels= mnist.test.labels,
                   prediction=prediction_result)

#可视化错误样本
def plot_images_labels_prediction(images,labels,prediction,
                                  index,num=10):
      fig = plt.gcf()
      fig.set_size_inches(10,12)
      if num > 25:
            num = 25
      for i in range(0,num):
          ax = plt.subplot(5,5,i+1)
          ax.inshow(np.reshape(images[index],(28,28)),
                    cmap='binary')
      title = "label=" +str(np.argmax(labels[index]))
      if len(prediction)>0:
          title +=",predict=" + str(prediction[index])

      ax.set_title(title,fontsize = 10)
      ax.set_xticks([])
      ax.set_yticks([])
      index +=1
    plt.show()





plot_images_labels_prediction(mnist.test.images,
                              mnist.test.labels,
                              prediction_result,610,20)

sess.close()
