

import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from sklearn.preprocessing import LabelEncoder
import pickle
import cv2 as cv
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



facenet = FaceNet()

faces_embeddings = np.load("face_embeddings_done.npz")
Y= faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

harrcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# model = pickle.loads(open("model.pkl","rb"))


def input_image(imgPath):
    inputImg = cv.imread(imgPath)
    inputImg = cv.cvtColor(inputImg,cv.COLOR_BGR2RGB)
    faces = harrcascade .detectMultiScale(inputImg,1.3,3)
    for (x,y,w,h) in faces:
        inputImg = inputImg[y:y+h,x:x+w]
        img = cv.resize(inputImg,(160,160))
        img = np.expand_dims(img, axis=0)
        with open('./model.pkl','rb') as p:
            model = pickle.load(p)
            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)
            name = encoder.inverse_transform(face_name)
            print('Name : ',name)

input_image("datasets/alex_carey/d600157cb7.jpg")




    