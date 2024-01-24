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

if __name__ == "__main__":
    flask_app.run(port=8001, debug=True)
