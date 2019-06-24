from HOF import HOF_frame, detect_face, face_cascade
import cv2
import numpy as np

EXPRESSIONS = ['Neutral', 'Happy', 'Sad', 'Surprised', 'Angry']
GENDER = ['Female', 'Male']
EXPRESSIONS_COLORS = {
    0: (207, 185, 151),
    1: (34, 139, 34),
    2: (255, 255, 255),
    3: (0, 0, 204),
    4: (204, 0, 0),
}

HOG = cv2.HOGDescriptor()


def load_svm_model(path):
    return cv2.ml.SVM_load(path)


def load_knn_model(path):
    knn = cv2.ml.KNearest_create()
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    knn_yml = fs.getNode('opencv_ml_knn')
    knn_format = knn_yml.getNode('format').real()
    is_classifier = knn_yml.getNode('is_classifier').real()
    default_k = knn_yml.getNode('default_k').real()
    samples = knn_yml.getNode('samples').mat()
    responses = knn_yml.getNode('responses').mat()
    fs.release()
    samples = np.array(samples, np.float32)
    knn.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return knn



def RecordedVideoTest(path, svm, genderClassifier, knn):
    cap = cv2.VideoCapture(path)
    ret, base = cap.read()
    base = detect_face(base)
    count = 0
    report = [0, 0, 0, 0, 0]
    gen = [0, 0]
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(face, 1.3, 5)

            count = count + 1
            for (x, y, w, h) in face:
                frame_face = frame[y:y + h, x:x + w]
                frame_facehog = cv2.resize(frame_face,(64, 128))
                frame_face = cv2.resize(frame_face, (128, 128))

                hog_data = HOG.compute(frame_facehog)
                hog_data = hog_data.reshape(1, hog_data.shape[0])
                gender = int(genderClassifier.predict(hog_data)[1][0])
                gen[gender] = gen[gender] + 1


                feature_vector = HOF_frame(base, frame_face)
                feature_vector = np.array([feature_vector], np.float32)
                expression = int(svm.predict(feature_vector)[1][0])
                report[expression] = report[expression] + 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), EXPRESSIONS_COLORS[expression], 2)

                cv2.putText(frame, (EXPRESSIONS[expression] + ' - ' + GENDER[gender]),
                            (x + w + 5, y + 5),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.7,
                            EXPRESSIONS_COLORS[expression],
                            2)

                cv2.imshow('Expression Detector', frame)
                base = frame_face
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except:
            break

    cap.release()
    cv2.destroyAllWindows()  # destroy all the opened windows
    print('******************(( Video Summary ))*********************')
    print('Number of tested frames : ', count)
    print('Neutral : ', report[0], ' (', ((report[0] / count) * 100), '%)')
    print('Happy : ', report[1], ' (', ((report[1] / count) * 100), '%)')
    print('Sad : ', report[2], ' (', ((report[2] / count) * 100), '%)')
    print('Surprised : ', report[3], ' (', ((report[3] / count) * 100), '%)')
    print('Angry : ', report[4], ' (', ((report[4] / count) * 100), '%)')
    print('Male : ', gen[1], ' (', ((gen[1] / count) * 100), '%)')
    print('Female : ', gen[0], ' (', ((gen[0] / count) * 100), '%)')
    print('**********************************************************')


if __name__ == "__main__":
    svm = load_svm_model('kohenmodel.dat')
    gender = load_svm_model("E:/kolya/year 4/semester 2/vision/project/hoggendersvmtest.dat")
    # knn = load_knn_model("HofKnnModel.yml")
    RecordedVideoTest("test//happy.mp4", svm, gender, None)
