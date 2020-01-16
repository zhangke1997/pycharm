import urllib.request
import os
import tarfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#导入数据集
def load_CIFAR_batch(filename):
    '''load single batch of cifar'''
    with open(filename,'rb')as f:
        data_dict =np.load(f,allow_pickle=True,encoding = 'bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']

        images =images.reshape(10000,3,32,32)
        images = images.transpose(0,2,3,1)
        labels = np.array(labels)
        return images,labels
def load_CIFAR_data(data_dir):
    '''load CIFAR data'''
    images_train = []
    labels_train = []
    for i in range(5):
        f = os.path.join(data_dir,'data_batch_%d' % (i+1))
        print('loading',f)
        image_batch,label_batch = load_CIFAR_batch(f)

        images_train.append(image_batch)
        labels_train.append(label_batch)
        Xtrain = np.concatenate(images_train)
        Ytrain = np.concatenate(labels_train)
        del image_batch,label_batch

    Xtest,Ytest = load_CIFAR_batch(os.path.join(data_dir,'test_batch'))
    print('finished loading CIFAR-10 data')

    return  Xtrain,Ytrain,Xtest,Ytest

data_dir = './cifar-10-batches-py'
Xtrain,Ytrain,Xtest,Ytest = load_CIFAR_data(data_dir)

'''
print('training data shape:',Xtrain.shape)
print('training labels shape:',Ytrain.shape)
print('test data shape:',Xtest.shape)
print('test labels shape:',Ytest.shape)
'''
"""
plt.imshow(Xtrain[9999])
print(Ytrain[9999])
plt.show()
"""


