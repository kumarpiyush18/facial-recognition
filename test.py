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
from scipy.spatial.distance import cosine
import pandas as pd

class TestSample:
    def __init__(self, directory,faces_embeddings_path,model_path):
        self.directory = directory
        self.facenet = FaceNet()
        self.faces_embeddings_path = faces_embeddings_path
        self.model_path = model_path
        self.encoder = LabelEncoder()
        self.harrcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.faces_embeddings = np.load(self.faces_embeddings_path) #"face_embeddings_done.npz"
    

    def calculate_confidence_scores(slef,recognized_embedding, known_embeddings):
        # Calculate cosine similarity between the recognized face and known identities
        # similarities = cosine_similarity([recognized_embedding], known_embeddings)
        # cosine_similarity = 1 - cosine(np.array([recognized_embedding]), np.array(known_embeddings))
        # known_embeddings = known_embeddings.
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
            with open(self.model_path,'rb') as p:
                model = pickle.load(p)
                ypred = self.facenet.embeddings(img)
                face_name = model.predict(ypred)
                label = self.encoder.inverse_transform(face_name)
            return imgPath,label,ypred[0]
    
    def test(self):
        Y,X = self.faces_embeddings['arr_1'], self.faces_embeddings['arr_0']
        known_embeddings = self.faces_embeddings['arr_0']
        known_labels = self.faces_embeddings['arr_1']
        Y = self.encoder.fit(Y)
        data= {
            "Image" : [],
            "Label": [],
            "Score": []
        }
        print("Test started ......on sample input")
        for files in os.listdir(self.directory):
            path = self.directory + '/' + files
            imgPath,label,ypred=self.input_image(path)
            score = self.calculate_confidence_scores(ypred,known_embeddings)
            data['Image'].append(imgPath)
            data['Label'].append(label[0])
            data['Score'].append(score)
            print(f"Filename: {imgPath}, Name : {label} , score {((score))*100}")
        df = pd.DataFrame(data)
        csv_file_path = './result.csv'
        df.to_csv(csv_file_path,index=False)
            
       

test_sample = TestSample("./output",faces_embeddings_path="./face_embeddings_done.npz",model_path="./model.pkl")
test_sample.test()