import cv2
import sys
import os
import time

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from faced import FaceDetector
from faced.utils import annotate_image



face_detector = FaceDetector()
smile_model = load_model('trained_model.h5')

cap = cv2.VideoCapture(0)

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

now = time.time()
while cap.isOpened():
    now = time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()

    if frame.shape[0] == 0:
        break

    bboxes = face_detector.predict(frame)
    gray_img = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    for x, y, h, w, _ in bboxes:
        crop = gray_img[int(y-h/2):int(y+h/2), int(x-h/3):int(x+h/3)]
        roi = cv2.resize(crop, (28, 28))
        ROI = roi.astype('float') / 255.0
        roi = img_to_array(ROI)
        roi = np.expand_dims(roi, axis=0)
        neutral, smile = smile_model.predict(roi)[0]
        label = 'Smiling' if smile > neutral else "Not Smiling"
        if label == "Smiling":
            cv2.putText(frame, "{:.4f}".format(smile)+" :)", (x-30, int(y+h/2 + 45)), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "{:.4f}".format(smile)+" :|", (x-30, int(y+h/2 + 45)), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 1)
    ann_frame = annotate_image(frame, bboxes)


    cv2.imshow('window', frame)

    print("FPS: {:0.2f}".format(1 / (time.time() - now)), end="\r", flush=True)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cv2.imshow('Image', ROI)
#cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
