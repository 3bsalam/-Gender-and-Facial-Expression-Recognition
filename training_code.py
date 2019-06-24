# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:34:06 2019

@author: Mohamed Adel
"""


import glob
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('D:\\haarcascade_frontalface_default.xml')



hog = cv2.HOGDescriptor()
data1=[]
labels1=[]


for f in glob.glob('D:/New_dataset/neutral/*.png'): #Neutral
    frame=cv2.imread(f)
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
    for(x,y,w,h) in faces:
        face = np.copy(frame[y:y+h, x:x+w])
    face=cv2.resize(face,(128,128))
    feat=hog.compute(face)
    data1.append(feat)
    labels1.append(0)   # 0 is for neutral
    #cv2.imshow('face',frame)
    
    
for f in glob.glob('D:/New_dataset/Happy/*.png'): #Happy
    frame=cv2.imread(f)
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
    for(x,y,w,h) in faces:
        face = np.copy(frame[y:y+h, x:x+w])
    face=cv2.resize(face,(128,128))
    feat=hog.compute(face)
    data1.append(feat)
    labels1.append(1)   # 1 is for Happy
    #cv2.imshow('face',frame)
    
    
    
    
    
for f in glob.glob('D:/New_dataset/sad/*.png'): #sad
    frame=cv2.imread(f)
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
    for(x,y,w,h) in faces:
        face = np.copy(frame[y:y+h, x:x+w])
    face=cv2.resize(face,(128,128))
    feat=hog.compute(face)
    data1.append(feat)
    labels1.append(2)   # 2 is for sad
    #cv2.imshow('face',frame)    
    
    
    
for f in glob.glob('D:/New_dataset/surprised/*.png'): #Surprised
    frame=cv2.imread(f)
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
    for(x,y,w,h) in faces:
        face = np.copy(frame[y:y+h, x:x+w])
    face=cv2.resize(face,(128,128))
    feat=hog.compute(face)
    data1.append(feat)
    labels1.append(3)   # 3 is for surprised
    #cv2.imshow('face',frame)    
    
    
for f in glob.glob('D:/New_dataset/angry/*.png'): #angry
    frame=cv2.imread(f)
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
    for(x,y,w,h) in faces:
        face = np.copy(frame[y:y+h, x:x+w])
    face=cv2.resize(face,(128,128))
    feat=hog.compute(face)
    data1.append(feat)
    labels1.append(4)   # 4 is for angry
   # cv2.imshow('face',frame)        
    
    
    
d=np.array(data1,np.float32)
l=np.array(labels1)
d=d.reshape(d.shape[0],d.shape[1]*d.shape[2])
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

print('started training')

svm.train(d, cv2.ml.ROW_SAMPLE, l)
print('finish')
svm.save('FinalHog3.dat')
print(len(data1))
print('Done')
    