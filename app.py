from HOF import HOF
import numpy as np
import cv2

train_vectors = []
train_labels = []

sad_ex = 's'
happy_ex = 'h'
neutral_ex = 'n'
suprised_ex = 'su'
angry_ex = 'A'

neutral_label = 0
happy_label = 1
sad_label = 2
suprised_label = 3
angry_label = 4


def train(path, label, expression, number_of_videos = 1):
    for i in range(0,number_of_videos):
        full_path = path + expression + str(i+1) + '.avi'
        tv, tl = HOF(full_path, label)
        for i in range(0, tv.shape[0]):
            train_vectors.append(tv[i])
            train_labels.append(label)
print('************************ Happy Videos ***************************')
train('videos//happy//', happy_label, happy_ex)
print('************************ Angry Videos ***************************')
train('videos//angry//', angry_label, angry_ex)

print('************************ Sad Videos ***************************')
train('videos//sad//', sad_label, sad_ex)
print('************************ Neutral Videos ***************************')
train('videos//neutral//', neutral_label, neutral_ex)
print('************************ Suprised Videos ***************************')
train('videos//suprised//', suprised_label, suprised_ex)

np.save('train_data', train_vectors)
np.save('train_labels', train_labels)





print('-----------------------------------------------------')
train_vectors = np.array(train_vectors, np.float32)
train_labels = np.array([train_labels]).T

print(train_vectors.shape)
print(train_labels)

print('=====================((Start Trainning))==================================')

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)

svm.train(train_vectors, cv2.ml.ROW_SAMPLE, train_labels)

knn = cv2.ml.KNearest_create()
knn.train(train_vectors, cv2.ml.ROW_SAMPLE, train_labels)

print('finish')
svm.save('kohenmodel.dat')
knn.save('kohenmodel.yml')
print('Done')
