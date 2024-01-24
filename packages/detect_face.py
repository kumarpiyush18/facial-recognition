import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from mtcnn.mtcnn import MTCNN


class FaceDetection:
    def __init__(self, directory):
        self.directory = directory
        self.targetSize = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    # extract face dimenstion using
    def extract_face_dim(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]['box']  # mtcnn face detection
        x, y = abs(x), abs(y)
        face = img[y:y + h, x:x + w]
        face_arr = cv2.resize(face, self.targetSize)
        return face_arr

    def load_images(self, dir):
        Faces = []
        for img_name in os.listdir(dir):
            try:
                path = dir + img_name
                single_face = self.extract_face_dim(path)
                Faces.append(single_face)
            except Exception as e:
                print("Error while processing image : ", img_name, "\n", e)

        return Faces

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory + '/' + sub_dir + '/'
            Faces = self.load_images(path)
            labels = [sub_dir for _ in range(len(Faces))]
            print(f"Loaded Successfully: {len(labels)}")
            self.X.extend(Faces)
            self.Y.extend(labels)
        return np.asarray(self.X), np.asarray(self.Y)
