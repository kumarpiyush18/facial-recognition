import pickle

import cv2
import flask
from flask import Response
from flask_cors import CORS

from backend.api.router import api
from backend.models import Base

from flask_sqlalchemy import SQLAlchemy

from packages.face_recognition import FaceRecognition

flask_app = flask.Flask(__name__)
flask_app.config.update(
    DEBUG=False,
    SQLALCHEMY_DATABASE_URI='postgresql+psycopg2://root:@127.0.0.1:5432/facial_recognition'
)
CORS(flask_app)
api.init_app(flask_app)

cap = cv2.VideoCapture(0)
search = FaceRecognition(directory="./sampleinput", faces_embeddings_path="./assets/face_embeddings_extract.npz", model_path="./model/SVC.model_small_class.pkl")


def gen():
    while True:
        _, frame = cap.read()
        processed_frame = search.draw_image(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        f = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n\r\n')


@flask_app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


def search_identity():
    return

if __name__ == "__main__":
    flask_app.run(port=8001, debug=True)
