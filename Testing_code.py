# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 06:04:07 2019

@author: Mohamed Adel
"""

import cv2 
import numpy as np

Arr=[]

Expre=['Neutral','Happy','Sad','Surprised','Angry']
hog = cv2.HOGDescriptor()
hoggendersvm=cv2.ml.SVM_load("FinalHog3.dat")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('Videodataset/suprised.mp4')

while cv2.waitKey(1) < 0:
    ret,frame = cap.read()   
    #frame=imutils.rotate(frame,90)
    if not ret:
        cv2.waitKey(0)
        cap.release()
        break
    
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
    for(x,y,w,h) in faces:
        face = np.copy(frame[y:y+h, x:x+w])
        
    face=cv2.resize(face,(128,128))
    hog_data=hog.compute(face)
    hog_data=hog_data.reshape(1,hog_data.shape[0])
    resg=hoggendersvm.predict(hog_data)
    Index=int(resg[1][0])
    EXP= Expre[Index]
    Arr.append(Index)
    cv2.rectangle(frame ,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.putText(frame,EXP,(x+w,y+5), cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
    cv2.imshow('frame',frame)
print(Arr)
    


    
    





