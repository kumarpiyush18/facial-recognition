import numpy as np

import os

from packages.app import get_embedding
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from sklearn.preprocessing import LabelEncoder
import pickle
import cv2
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



detector = MTCNN()
def input_image(imgPath):
    inputImg = cv2.imread(imgPath)
    inputImg = cv2.cvtColor(inputImg,cv2.COLOR_BGR2RGB)
    x,y,w,h = detector.detect_faces(inputImg)[0]['box']
    inputImg = inputImg[y:y+h,x:x+w]
    inputImg = cv2.resize(inputImg,(160,160))
    test_img = get_embedding(inputImg)
    test_img = [test_img]
    print("load pickel file")
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    ypreds = pickled_model.predict(test_img)
    print("prediction completed")
    # ypreds = model.predict(test_img)
    # print(encoder.inverse_transform(ypreds))
    print(ypreds)


input_image("datasets/adil_rashid/e14dbc0627.jpg")
