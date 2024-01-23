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


class FaceRecognition:
    def __init__(self,directory):
        self.directory = directory
        self.targetSize = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face_dim(self,filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x),abs(y)
        face = img[y:y+h,x:x+w]
        face_arr = cv2.resize(face,self.targetSize)
        return face_arr
    
    def load_faces(self,dir):
        Faces = []
        for img_name in os.listdir(dir):
            try:
                path = dir + img_name
                single_face = self.extract_face_dim(path)
                Faces.append(single_face)
            except Exception as e:
                print("Error while processing image : ",img_name,"\n",e)

        return Faces
    
    def load_classes(self):
        for sub_dir  in os.listdir(self.directory):
            path = self.directory + '/' +sub_dir + '/'
            Faces = self.load_faces(path)
            labels = [sub_dir for _ in range(len(Faces))]
            print(f"Loaded Successfully: {len(labels)}")
            self.X.extend(Faces)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)
    



faceloading = FaceRecognition("./datasets")
X,Y =faceloading.load_classes()

embeder = FaceNet()
def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img,axis=0)
    yhat = embeder.embeddings(face_img)
    return yhat[0]


Embedded_X = []
for img in X:
    Embedded_X.append(get_embedding(img))

Embedded_X = np.asarray(Embedded_X)
np.savez_compressed('./face_embeddings_done.npz',Embedded_X,Y)


encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
X_train,X_test,y_train,y_test = train_test_split(Embedded_X,Y, test_size=0.3, shuffle=True,random_state=0)

# print(Embedded_X.shape)
# Embedded_X = Embedded_X.transpose()

# print(Y.shape)

model = SVC(kernel='linear',probability=True)
model.fit(X_train,y_train)
pickle.dump(model, open('./model.pkl', 'wb'))


ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)



try:
    accuracy_score(y_train,ypreds_test)
    accuracy_score(y_test,ypreds_test)
except Exception as e:
    print(f"Error in calculating the accuracy score : {e}")




# detector = MTCNN()
# def input_image(imgPath):
#     inputImg = cv2.imread(imgPath)
#     inputImg = cv2.cvtColor(inputImg,cv2.COLOR_BGR2RGB)
#     x,y,w,h = detector.detect_faces(inputImg)[0]['box']
#     inputImg = inputImg[y:y+h,x:x+w]
#     inputImg = cv2.resize(inputImg,(160,160))
#     test_img = get_embedding(inputImg)
#     test_img = [test_img]
#     print("load pickel file")
#     pickled_model = pickle.load(open('model.pkl', 'rb'))
#     ypreds = pickled_model.predict(test_img)
#     print("prediction completed")
#     ypreds = model.predict(test_img)
#     print(encoder.inverse_transform(ypreds))
    


# input_image("datasets/adil_rashid/e14dbc0627.jpg")
