import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def smile(face_array):
    smile_model = load_model('trained_model.h5')
    roi = cv2.resize(face_array, (28, 28))
    ROI = roi.astype('float') / 255.0
    roi = img_to_array(ROI)
    roi = np.expand_dims(roi, axis=0)
    _, smile_intensity = smile_model.predict(roi)[0]
return smile_intensity
