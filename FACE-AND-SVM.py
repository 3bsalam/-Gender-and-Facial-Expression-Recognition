import cv2
import numpy as np


Gender=['Female','Male']
hog = cv2.HOGDescriptor()
hoggendersvm=cv2.ml.SVM_load("E:/kolya/year 4/semester 2/vision/project/hoggendersvmtest.dat")

face_cascade = cv2.CascadeClassifier('E:/kolya/year 4/semester 2/vision/project/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('E:/kolya/year 4/semester 2/vision/project/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
    for(x,y,w,h) in faces:
        face = np.copy(img[y:y+h, x:x+w])
        face= cv2.resize(face, (64, 128))
        hog_data=hog.compute(face)
       ## print("abl",hog_data)
        hog_data=hog_data.reshape(1,hog_data.shape[0])
       ## print("b3d",hog_data)
        resg=hoggendersvm.predict(hog_data)
        #print("el resg",resg)
        Index=int(resg[1][0])
        gen= Gender[Index]
        print(Index)
        cv2.rectangle(img ,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,gen,(x+w,y+5), cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
        roi_gray = gray[ y:int((y+h)*0.8),x:x+w]
        roi_color= img[ y:int((y+h)*0.8),x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh), (0,255,0),2)
            
            
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows() 