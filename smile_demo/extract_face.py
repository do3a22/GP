import numpy as np
import cv2
from faced.detector import FaceDetector

def extract(img):
	crop=np.array([])
	face_detector = FaceDetector()
	bboxes = face_detector.predict(img)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for x, y, h, w, _ in bboxes:
		crop = gray_img[int(y-h/2):int(y+h/2), int(x-h/3):int(x+h/3)]
	return crop
