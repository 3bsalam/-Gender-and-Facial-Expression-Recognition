# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:54:22 2019

@author: Mohamed Adel
"""

import glob
import numpy as np
import cv2


hog = cv2.HOGDescriptor()
data1=[]
labels1=[]

for f in glob.glob('E:/kolya/year 4/semester 2/vision/project/Female/*.jpg'):
    img=cv2.imread(f)
    print('Female')
    img=cv2.resize(img,(64,128))
    feat=hog.compute(img)
    data1.append(feat)
    labels1.append(0)


for f in glob.glob('E:/kolya/year 4/semester 2/vision/project/Male/*.jpg'):
    img=cv2.imread(f)
    print('Male')
    img=cv2.resize(img,(64,128))
    feat=hog.compute(img)
    data1.append(feat)
    labels1.append(1)


d=np.array(data1,np.float32)
l=np.array(labels1)
d=d.reshape(d.shape[0],d.shape[1]*d.shape[2])
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

print('started training')

svm.train(d, cv2.ml.ROW_SAMPLE, l)
print('finish')
svm.save('hoggendersvmtest.dat')
print(len(data1))
print('Done')
