import matplotlib.pyplot as plt
from date import Xtrain,Ytrain,Xtest,Ytest

label_dict = {0:"airplane",1:"automobile",2:"bird",3:"cat",
              4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,6)
    if num>10:
        num=10
    for i in range(0,num):
        ax = plt.subplot(2,5,1+i)
        ax.imshow(images[idx],cmap = 'binary')

        title = str(i)+','+label_dict[labels[idx]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[idx]]
        ax.set_title(tile,fontsize =10)

        idx+=1
    plt.show()

plot_images_labels_prediction(Xtest,Ytest,[],1,10)