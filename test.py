import tensorflow as tf
from tensorflow.keras.callbacks import tensorboard
tf.reset_default_graph()

logdir='/home/zk/PycharmProjects/firstone'

input1 = tf.constant([1,2,3],name = "input1")
input2 = tf.constant(tf.random_uniform([3],name = "input2"))
output = tf.add_n([input1,input2],name = "add")
writer = tf.summary.FileWriter(logdir,tf.get_default_grapt())
writer.close()