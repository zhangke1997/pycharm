import tensorflow as tf
'''
#(tf.__version__)
node1 = tf.constant(3.0,tf.float32,name="node1")
node2 = tf.constant(4.0,tf.float32,name="node2")
node3 = tf.add(node1,node2)

#print(node3)//输出的式张量 一个计算过程
sess=tf.Session()
print(sess.run(node1),sess.run(node2),sess.run(node3))#run相当于吧张量变成python变量
sess.close()



tens1=tf.constant([[[1, 2, 2], [2, 2, 3]],
                   [[3, 5, 6], [5, 4, 3]],
                   [[7, 0, 1], [9, 1,9]]])
print(tens1)#Tensor("Const:0", shape=(3, 2, 3), dtype=int32)
print(tens1.get_shape()) #维度有大到小显示
'''

tf.reset_default_graph()#清楚之前的
a = tf.Variable(1, name="a")
b = tf.add(a, 1, name="b")
c = tf. multiply(b, 4, name="c")
d = tf.subtract(c, b, name="d")
##
logdir = '/home/zk/PycharmProjects/second'
writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
writer.close()

##会话1
sess = tf.Session()
try:
    print(sess.run(tens1))
except:
    print("Exception")
finally:
    sess.close()

#会话2 自动关闭
with tf.Session() as sess:
    print(sess.run(tens1))
# 会话3 设置默认会话,获取张量的值
sess = tf.InteractiveSession()
print(b.eval())#没有上局要指定会话print(b.eval(session=sess))
sess.close()

#变量要初始化,常量不用