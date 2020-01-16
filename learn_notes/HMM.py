import numpy as np
import  math as mt
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('lena.jpg')
cluster_num = 4;%设置分类数
maxiter = 60;%最大迭代次数
#-------------随机初始化标签----------------
label = np.random.randint([1,cluster_num],Image.size(img))
#-----------kmeans最为初始化预分割----------
# label = kmeans(img(:),cluster_num)
#label = reshape(label,size(img))
iter = 0;
while iter < maxiter:
#    %-------计算先验概率---------------
#    %这里我采用的是像素点和3*3领域的标签相同
#    %与否来作为计算概率
#   %------收集上下左右斜等八个方向的标签--------
    label_u = .imfilter(label,[0,1,0;0,0,0;0,0,0],'replicate');
    label_d = imfilter(label,[0,0,0;0,0,0;0,1,0],'replicate');
    label_l = imfilter(label,[0,0,0;1,0,0;0,0,0],'replicate');
    label_r = imfilter(label,[0,0,0;0,0,1;0,0,0],'replicate');
    label_ul = imfilter(label,[1,0,0;0,0,0;0,0,0],'replicate');
    label_ur = imfilter(label,[0,0,1;0,0,0;0,0,0],'replicate');
    label_dl = imfilter(label,[0,0,0;0,0,0;1,0,0],'replicate');
    label_dr = imfilter(label,[0,0,0;0,0,0;0,0,1],'replicate');
    p_c = zeros(4,size(label,1)*size(label,2));
#    %计算像素点8领域标签相对于每一类的相同个数
    for i = 1:cluster_num
        label_i = i * ones(size(label))
        temp = ~(label_i - label_u) + ~(label_i - label_d) +
            (label_i - label_l) + ~(label_i - label_r) +
            (label_i - label_ul) + ~(label_i - label_ur) + ...
            (label_i - label_dl) +~(label_i - label_dr)
        p_c(i,:) = temp(:)/8  #计算概率

    p_c(find(p_c == 0)) = 0.001 #防止出现0
#    %---------------计算似然函数----------------
    mu = np.zeros(1,4)
    sigma = np.zeros(1,4)
#    %求出每一类的的高斯参数--均值方差
    for i in range(1,cluster_num):
        index = find(label == i)#找到每一类的点
        data_c = img(index)
        mu(i) = mean(data_c)#%均值
        sigma(i) = var(data_c)#方差
    end
    p_sc = zeros(4,size(label,1)*size(label,2))
#     for i = 1:size(img,1)*size(img,2)
#         for j = 1:cluster_num
#             p_sc(j,i) = 1/sqrt(2*pi*sigma(j))*...
#               exp(-(img(i)-mu(j))^2/2/sigma(j));
#         end
#     end
#------计算每个像素点属于每一类的似然概率--------
#------为了加速运算，将循环改为矩阵一起操作--------
    for j in  range(1,cluster_num):
        MU = repmat(mu(j),size(img,1)*size(img,2),1)
        p_sc(j,:) = 1/math.sqrt(2*pi*sigma(j))*...
            exp(-(img(:)-MU).^2/2/sigma(j))
    end
    %找到联合一起的最大概率最为标签，取对数防止值太小
    [~,label] = max(log(p_c) + log(p_sc))
    %改大小便于显示
    label = reshape(label,size(img))
#---------显示----------------
    if ~mod(iter,6)
        figure
        n=1
    end
    subplot(2,3,n)
    imshow(label,[])
    title(['iter = ',num2str(iter)])
    pause(0.1)
    n = n+1
    iter = iter + 1
