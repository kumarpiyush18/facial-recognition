import json
import tempfile
import traceback

import cv2
from flask import request, Response
from flask_restful import Resource

from packages.face_recognition import FaceRecognition

from http import HTTPStatus


class ApiResponse:
    def __init__(self, success: bool = False, message: str = '', error: dict = None, data=None, meta=None):
        self.success = success
        self.error = error
        self.message = message
        self.data = {} if data is None else data
        self.meta = {} if meta is None else meta

    def json(self):
        return json.dumps(self.__dict__)


class HealthCheck(Resource):
    def get(self):
        return 'Hello World!!'


cap = cv2.VideoCapture(0)
search = FaceRecognition(directory="./sampleinput", faces_embeddings_path="./assets/face_embeddings_extract.npz",
                         model_path="./model/SVC.model_small_class.pkl", outputdir='./output')


def gen():
    while True:
        _, frame = cap.read()
        processed_frame = search.draw_image(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        f = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n\r\n')


class VideoFeed(Resource):

    def get(self):
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


class SearchIdentity(Resource):

    def post(self):

        request_id = request.form.get('request_id')
        file = request.files.get('input')
        if request_id == '':
            resp = ApiResponse(success=False, message='empty request_id')
            return Response(response=resp.json(), status=HTTPStatus.BAD_REQUEST, mimetype="application/json")
        if file is None:
            resp = ApiResponse(success=False, message='input file expected')
            return Response(response=resp.json(), status=HTTPStatus.BAD_REQUEST, mimetype="application/json")

        filename = file.filename

        # image file only
        is_image = (filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"))
        if not is_image:
            resp = ApiResponse(success=False, message='only image file is expected')
            return Response(response=resp.json(), status=HTTPStatus.BAD_REQUEST, mimetype="application/json")
        try:
            # generate temp file
            tmp_file = tempfile.NamedTemporaryFile().name + ".jpeg"
            print('temp file path', tmp_file)
            file.save(tmp_file)

            input = cv2.imread(tmp_file)
            label, score = search.search_identity(input)
            if label == '' or score == '' or score == '0':
                resp = ApiResponse(
                    success=False,
                    message='No Person Found!',
                )
                return Response(response=resp.json(), status=HTTPStatus.OK, mimetype="application/json")

            resp = ApiResponse(
                success=True,
                message='Identity Found!',
                data={
                    "label": label,
                    "score": score,
                }
            )
        except Exception as e:
            resp = ApiResponse(
                success=False,
                error={
                    'error_msg': getattr(e, 'message', repr(e)),
                    'stack_trace': traceback.format_exc()
                }
            )

            return Response(response=resp.json(), status=HTTPStatus.INTERNAL_SERVER_ERROR, mimetype="application/json")

        return Response(response=resp.json(), status=HTTPStatus.OK, mimetype="application/json")
