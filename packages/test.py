import os

import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pickle
import cv2 as cv
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class TestSample:
    def __init__(self, directory,faces_embeddings_path,model_path):
        self.directory = directory
        self.facenet = FaceNet()
        self.faces_embeddings_path = faces_embeddings_path
        self.model_path = model_path
        self.encoder = LabelEncoder()
        self.harrcascade = cv.CascadeClassifier("./assets/haarcascade_frontalface_default.xml")
        self.faces_embeddings = np.load(self.faces_embeddings_path) #"face_embeddings_done.npz"
        with open(self.model_path,'rb') as p:
                self.model = pickle.load(p)
        self.Y,self.X = self.faces_embeddings['arr_1'], self.faces_embeddings['arr_0']
        self.encoder.fit(self.Y)
    

    def calculate_confidence_scores(slef,recognized_embedding, known_embeddings):

        # calculate similarity
        cosine_similarity = 1-np.dot([recognized_embedding],known_embeddings.T)/( np.linalg.norm([recognized_embedding]) * np.linalg.norm(known_embeddings,axis=1))
        confidence_score = np.mean(cosine_similarity)
        return confidence_score

    def input_image(self,imgPath):
        inputImg = cv.imread(imgPath)
        inputImg = cv.cvtColor(inputImg,cv.COLOR_BGR2RGB)
        faces = self.harrcascade.detectMultiScale(inputImg,1.3,2)
        for (x,y,w,h) in faces:
            inputImg = inputImg[y:y+h,x:x+w]
            img = cv.resize(inputImg,(160,160))
            img = np.expand_dims(img, axis=0)
            ypred = self.facenet.embeddings(img)
            face_name = self.model.predict(ypred)
            label = self.encoder.inverse_transform(face_name)
            return imgPath,label,ypred[0]

    def draw_image(self,frame):
        inputImg = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        faces = self.harrcascade.detectMultiScale(inputImg,1.3,2)
        print('face detected in', faces)
        try:
            for (x,y,w,h) in faces:
                inputImg = inputImg[y:y+h,x:x+w]
                img = cv.resize(inputImg,(160,160))
                img = np.expand_dims(img, axis=0)
                ypred = self.facenet.embeddings(img)
                face_name = self.model.predict(ypred)
                label = self.encoder.inverse_transform(face_name)
                score = self.calculate_confidence_scores(ypred, self.X)

                if score > 0.92:
                    frame = cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)
                    frame = cv2.putText(frame, "{} ({})".format(label, score), (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 255), 3)
        except Exception as e:
            print(e)
        return frame

    def run(self):
        data= {
            "Image" : [],
            "Label": [],
            "Score": []
        }
        print("Test started ......on sample input")
        for files in os.listdir(self.directory):
            path = self.directory + '/' + files
            imgPath,label,ypred=self.input_image(path)
            score = self.calculate_confidence_scores(ypred,self.X)
            data['Image'].append(imgPath)
            data['Label'].append(label[0])
            data['Score'].append(score)
            print(f"Filename: {imgPath}, Name : {label[0]} , score {((score))*100}")
        df = pd.DataFrame(data)
        csv_file_path = './output/result.csv'
        df.to_csv(csv_file_path,index=False)


    def search_identity(self, img):
        pass

