import pandas as pd
import numpy as np

df = pd.read_csv('11111.csv',encoding="utf-8")
print(df.describe())
df = np.array(df)
print(df.shape)
x_data = df[:, :4]#前十二列
y_data = df[:, 5]#第十二列
x_data = np.array(x_data)
y_data =np.array(y_data)
print(x_data)
print(y_data)
for i in range(1, 6):
    df[:,i] = df[:,i]/(df[:,i].max()-df[:,i].min())

x_data = df[:, :4]#前十二列
y_data = df[:, 5]#第十二列
x_data = np.array(x_data)
y_data =np.array(y_data)
print(x_data)
print(y_data)
for i in range(5):
    print(i)
