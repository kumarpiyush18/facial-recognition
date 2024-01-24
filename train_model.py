import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
from keras_facenet import FaceNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from packages.detect_face import FaceDetection
from sklearn.neighbors import KNeighborsClassifier

import sys
import os

if hasattr(sys, '_MEIPASS'):
    # Running as a PyInstaller bundled executable
    base_path = sys._MEIPASS
else:
    # Running as a script
    base_path = os.getcwd()

class Train:
    def __init__(self, X, Y):
        self.embedder = FaceNet()
        self.X = X
        self.Y = Y
        self.encoder = LabelEncoder()
        self.Embedded_X = []

    def get_face_embeddings(self, face_imgs):
        face_imgs = face_imgs.astype('float32')
        face_imgs = np.expand_dims(face_imgs, axis=0)
        yhat = self.embedder.embeddings(face_imgs)
        return yhat[0]

    def prepare_data(self):
        print("Preparing Face embeddings......")
        for img in self.X:
            self.Embedded_X.append(self.get_face_embeddings(img))
        self.Embedded_X = np.asarray(self.Embedded_X)
        try:
            np.savez_compressed(os.path.join(base_path,"face_embeddings_extract.npz"), self.Embedded_X, self.Y)
        except Exception as e:
            print("Some Problem occured while saving to the file : ", e)
        print("Face embeddings completed......")

    def train_data(self):
        print("Model training start.........")
        self.encoder.fit(self.Y)
        self.Y = self.encoder.transform(self.Y)
        X_train, X_test, y_train, y_test = train_test_split(self.Embedded_X, self.Y, test_size=0.2, shuffle=True,
                                                            random_state=0)
        model = SVC(kernel='linear', probability=True)
        knn_model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        knn_model.fit(X_train,y_train)
        try:
            # pickle.dump(model, open(os.path.join(get_base_path(),"model","SVC.model_small_class.pkl"), 'wb'))
            pickle.dump(model, open(os.path.join(base_path,"model","Knn.model_class.pkl"), 'wb'))
            # svc_ypreds_test = model.predict(X_test)
            knn_ypreds_test = knn_model.predict(X_test)
            # svc_acc = accuracy_score(y_test, svc_ypreds_test)
            knn_acc = accuracy_score(y_test, knn_ypreds_test)

        except Exception as e:
            print(f"Error in calculating the accuracy score : {e}")
        print("Model training completed.........")
        # print(f"Accuracy of SVC: {svc_acc * 100:.2f}%")
        print(f"Accuracy of KNN: {knn_acc * 100:.2f}%")


if __name__ == "__main__":
    facedetect = FaceDetection(os.path.join(base_path,"datasets"))
    X, Y = facedetect.load_classes()

    train = Train(X, Y)
    train.prepare_data()
    train.train_data()
