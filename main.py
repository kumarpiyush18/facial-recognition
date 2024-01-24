from packages.detect_face import FaceDetection
from packages.train_model import Train

facedetect = FaceDetection("./datasets")
X,Y = facedetect.load_classes()

train = Train(X,Y)
train.prepare_data()
train.train_data()
