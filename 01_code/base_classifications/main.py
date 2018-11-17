#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "chen"

import time
import cv2 as cv
from PIL import Image
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
# from keras.models import load_model
# from skimage.io import imshow, imread, imsave
import auxiliary



if __name__ == '__main__':

    md = auxiliary.Mouth_Decector()

    # model = load_model('./model_by_nn_48-28_acc091.h5')
    model = joblib.load('./68pts_svm.joblib')

    cv.namedWindow('hello_smile')

    vc = cv.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    cap = cv.VideoCapture(0)

    print('start capature video')

    fps = 120
    width = 48 # 28
    height = 28 # 10

    while True:
        start_time = time.time()

        _, img_org = cap.read()

        mouth = md.find_mouth(img_org, is_square_face=True, is_square_mouth=True)

        if (len(mouth) > 0):

            x, y, w, h = mouth
            mouth_img = md.get_partial(img_org, x, y, w, h)

            # imsave(mouth_img, './my_face')
            mouth_vec = md.normalize_img(mouth_img, is_vectorize=True, width=width, height=height)
            # data_img_vec_norm = normalize(mouth_vec, norm='l2', axis=1)
            # data_img_vec_norm = data_img_vec_norm.reshape(-1, height, width, 1)
            data_img_vec_norm = (mouth_vec / 255).reshape(1,-1)

            result = model.predict(data_img_vec_norm)
            is_smile = result[0] == 1

            cv.putText(img_org, 
                       'smile' if is_smile else 'not smile',  
                       (x, y - 5), 
                       cv.FONT_HERSHEY_PLAIN, 
                       2, 
                       (0,0,255))

            print(f"{mouth}, is_smile: {is_smile}, smile_prop: {result[0]}")

        cv.imshow("looking", img_org)

        time.sleep(1.0 / fps)

        if (cv.waitKey(10) & 0xFF == ord('q')):
            cap.release()
            cv.destroyallwindows()
            break
