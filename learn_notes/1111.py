import tensorflow as tf
import keras
from keras import layers

# 构建一个顺序模型
model = Sequential()
# 模型中添加一个全连接层
model.add(Dense(units=1, input_dim=1))
# sgd随机梯度下降，mse均方误差
model.compile(optimizer='sgd', loss='mse')

# 训练3001个批次
for step in range(3001):
    # 每次训练一个批次
    cost = model.train_on_batch(x_data, y_data)
    if step % 500 == 0:
        print('cost:', cost)

# 打印权值和偏置值
w, b = model.layers[0].get_weights()
print('w:', w, 'b:', b)

# x_data输入网络中，得到预测值y_pred
y_pred = model.predict(x_data)
