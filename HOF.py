import cv2
import numpy as np
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
hsv = np.zeros((8, 8, 3))
hsv[..., 1] = 255
histobin = [0, 20, 40, 60, 80, 100, 120, 140, 160]


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    result = ''
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (128, 128))
        result = face
    return result


def angle_exist(number):
    if number == 180:
        return 0, True

    for idx, val in enumerate(histobin):
        if val == number:
            return idx, True
    return -1, False


def get_indicies(number):
    min = 0
    max = 0

    if number > 160:
        min = 160
        max = 180
        return min, max

    # check max
    for i in range(0, 160, 20):
        if number > i:
            min = i
            max = i + 20

    return min, max


def get_hog(mag, ang):
    # initialization
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for x in range(0, ang.shape[1]):
        for y in range(0, ang.shape[0]):
            num = ang[x, y]
            indx, exist = angle_exist(num)
            if exist:
                result[indx] = result[indx] + mag[x, y]
            else:
                min, max = get_indicies(num)
                indx1, e = angle_exist(min)
                indx2, e1 = angle_exist(max)
                precentMax = (num - min) / (max - min)
                precentMin = 1 - precentMax
                val = mag[x, y]
                result[indx1] = result[indx1] + (val * precentMin)
                result[indx2] = result[indx2] + (val * precentMax)

    return result


def window_analysis(img, base):
    stepSize = 8
    result = []
    (w_width, w_height) = (8, 8)
    for x in range(0, img.shape[1], stepSize):
        result_row = []
        for y in range(0, img.shape[0], stepSize):
            window_1 = base[x:x + w_width, y:y + w_height]
            window_2 = img[x:x + w_width, y:y + w_height]
            # print(len(window_1))
            flow = cv2.calcOpticalFlowFarneback(window_1, window_2, None, 0.5, 3, 15, 3, 5, 1.2,
                                                0)  # pyr_scale,levels,winsize,iter,poly
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            result_row.append(get_hog(hsv[..., 2], hsv[..., 0]))
            # print(get_hog(hsv[..., 2],hsv[..., 0]))
            # print('--------------------------------------------------------------------------------------------')
            # print('angle /n')
            # print(hsv[..., 0])
            # print('mag /n')
            # print(hsv[...,2])
            # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result.append(result_row)

    return result


"""
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
############################################################ Frames Normalization #################################################################################################### 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


def calculate_vector_square(vec):
    sum = 0
    for i in range(0, len(vec) - 1):
        sum = sum + (vec[i] * vec[i])
    return sum


def get_norm_avg(window):
    sum = 0
    for x in range(0, window.shape[1]):
        for y in range(0, window.shape[0]):
            vec = window[x, y]
            sum = sum + calculate_vector_square(vec)
    return math.sqrt(sum)


def frame_analysis(frame):
    stepSize = 1
    result = []
    # print(frame.shape)
    (w_width, w_height) = (2, 2)
    for x in range(0, frame.shape[1] - 1, stepSize):
        for y in range(0, frame.shape[0] - 1, stepSize):
            window = frame[x:x + w_width, y:y + w_height]
            norm = get_norm_avg(window)
            for xx in range(0, window.shape[1]):
                for yy in range(0, window.shape[0]):
                    for zz in range(0, window.shape[2]):
                        result.append((window[xx, yy, zz] / norm))
    return result


def HOF(videopath, label):
    cap = cv2.VideoCapture(videopath)
    ret, frame1 = cap.read()
    base = detect_face(frame1)
    base = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    non_normilized_frame = []
    stop = 0

    while True:
        try:
            if stop == 1000:
                break
            if stop % 1 == 0:
                ret, img = cap.read()
                img = detect_face(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result_frame = window_analysis(img, base)
                non_normilized_frame.append(result_frame)
                print('Proccessing frame number : ', len(non_normilized_frame))
                base = img
                stop = stop + 1
            else:
                stop = stop + 1

        except:
            break

    train_vectors = []

    for i in range(0, len(non_normilized_frame)):
        n_frame = non_normilized_frame[i]
        n_frame = np.array(n_frame)
        train_vectors.append(frame_analysis(n_frame))

    train_vectors = np.array(train_vectors)
    train_labeles = []

    for i in range(0, len(train_vectors)):
        train_labeles.append(label)

    train_labeles = np.array([train_labeles]).T

    print('--------------------------------------------------------------------------------')
    print('feature vector dimensions: ', train_vectors.shape)
    # print('labels :', train_labeles)

    cap.release()
    cv2.destroyAllWindows()

    return train_vectors, train_labeles


def HOF_frame(base, frame):
    base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result_frame = window_analysis(img, base)

    n_frame = np.array(result_frame)
    feature_vector = frame_analysis(n_frame)

    feature_vector = np.array(feature_vector)

    return feature_vector
