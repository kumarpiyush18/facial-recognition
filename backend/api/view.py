import traceback
from http import HTTPStatus

import cv2
import flask
import psycopg2
from flask import request
from flask_restful import Resource


class HealthCheck(Resource):
    def get(self):
        return 'Hello World!!'


class FaceMatch(Resource):
    def post(self):
        return ""
