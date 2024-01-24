
import numpy as np

import os
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

class Train:
    def __init__(self,X,Y):
        self.embedder = FaceNet()
        self.X =X
        self.Y =Y
        self.encoder=LabelEncoder()
        self.Embedded_X = []
    def get_face_embeddings(self,face_imgs):
        face_imgs = face_imgs.astype('float32')
        face_imgs = np.expand_dims(face_imgs,axis=0) 
        yhat = self.embedder.embeddings(face_imgs)
        return yhat[0]
    
    def prepare_data(self):
        print("Preparing Face embeddings......")
        for img in self.X:
            self.Embedded_X.append(self.get_face_embeddings(img))
        self.Embedded_X = np.asarray(self.Embedded_X)
        try:
            np.savez_compressed('./face_embeddings_extract.npz',self.Embedded_X,self.Y)
        except Exception as e:
            print("Some Problem occured while saving to the file : ",e)
        print("Face embeddings completed......")

    def train_data(self):
        print("Model training start.........")
        self.encoder.fit(self.Y)
        self.Y = self.encoder.transform(self.Y)
        X_train,X_test,y_train,y_test = train_test_split(self.Embedded_X,self.Y, test_size=0.2, shuffle=True,random_state=0)
        model = SVC(kernel='linear',probability=True)
        model.fit(X_train,y_train)
        try:
            pickle.dump(model, open('./model/SVC.model_small_class.pkl', 'wb'))
            ypreds_test = model.predict(X_test)
            acc = accuracy_score(y_test,ypreds_test)
        except Exception as e:
            print(f"Error in calculating the accuracy score : {e}")
        print("Model training completed.........")
        print(f"Accuracy: {acc * 100:.2f}%")
        


