from HOF import HOF , HOF_frame
import numpy as np
import cv2


def load_knn(path):
    knn = cv2.ml.KNearest_create()
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    knn_yml = fs.getNode('opencv_ml_knn')
    knn_format = knn_yml.getNode('format').real()
    is_classifier = knn_yml.getNode('is_classifier').real()
    default_k = knn_yml.getNode('default_k').real()
    samples = knn_yml.getNode('samples').mat()
    responses = knn_yml.getNode('responses').mat()
    fs.release()
    samples = np.array(samples,np.float32)
    knn.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return knn


Arr=[]
Expre=['Neutral','Happy','Sad','Surprised','Angry']
hofsvm=cv2.ml.SVM_load("kohenmodel.dat")
hofknn = load_knn("kohenmodel.yml")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = "test//su3.mp4"

neutral_label = 0
happy_label = 1
sad_label = 2
suprised_label = 3
angry_label = 4



hof_data, A = HOF(path,-1)
hof_data = np.array(hof_data, np.float32)

# svm test
resg = hofsvm.predict(hof_data)
print('predicted frames:' , len(resg[1]))
print(resg[1])

accuracy = 0
for i in range (0, len(resg[1])):
    if resg[1][i] == suprised_label:
        accuracy = accuracy + 1

overall_accuracy = ((accuracy) / (len(resg[1]))) * 100
print('SVM overall Accuracy : ', overall_accuracy)
# np.save('testing', hof_data)

# KNN test
ret, predict, neighbours, dist = hofknn.findNearest(hof_data, k=5)
print('predicted frames:', len(predict))
print(predict)

accuracy = 0
for i in range(0, len(predict)):
    if predict[i] == suprised_label:
        accuracy = accuracy + 1

overall_accuracy = ((accuracy) / (len(predict))) * 100
print('KNN overall Accuracy : ', overall_accuracy)



cap = cv2.VideoCapture(path)
ret, frame1 = cap.read()

# while cv2.waitKey(1) < 0:
#     ret, frame = cap.read()
#     # frame=imutils.rotate(frame,90)
#     if not ret:
#         cv2.waitKey(0)
#         cap.release()
#         break
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     hof_data = HOF_frame(frame1,frame)
#     hof_data = np.array(hof_data, np.float32)
#     for (x, y, w, h) in faces:
#         face = np.copy(frame[y:y + h, x:x + w])
#         hof_data.reshape(1, hof_data.shape[0])
#         resg = hofsvm.predict(hof_data)
#         Index = int(resg[1][0])
#         EXP = Expre[Index]
#         Arr.append(Index)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(frame, EXP, (x + w, y + 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.imshow('frame', frame)
#     frame1 = frame
#
# print(Arr)