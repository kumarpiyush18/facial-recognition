

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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


def calculate_confidence_scores(recognized_embedding, known_embeddings):
    # Calculate cosine similarity between the recognized face and known identities
    similarities = cosine_similarity([recognized_embedding], known_embeddings)
    confidence_score = np.mean(similarities)
    return confidence_score

facenet = FaceNet()

faces_embeddings = np.load("face_embeddings_done.npz")

Y= faces_embeddings['arr_1'] # label
X= faces_embeddings['arr_0']  #embeddings

known_embeddings = faces_embeddings['arr_0']
known_labels = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
harrcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# model = pickle.loads(open("model.pkl","rb"))


def input_image(imgPath):
    inputImg = cv.imread(imgPath)
    inputImg = cv.cvtColor(inputImg,cv.COLOR_BGR2RGB)
    faces = harrcascade .detectMultiScale(inputImg,1.3,2)
    for (x,y,w,h) in faces:
        inputImg = inputImg[y:y+h,x:x+w]
        img = cv.resize(inputImg,(160,160))
        img = np.expand_dims(img, axis=0)
        with open('./model.pkl','rb') as p:
            model = pickle.load(p)
            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)
            label = encoder.inverse_transform(face_name)
            score = calculate_confidence_scores(ypred[0],known_embeddings)
            print(f"Filename: {imgPath}, Name : {label} , score {1-score}")

input_image("testInput/1.jpg")




    