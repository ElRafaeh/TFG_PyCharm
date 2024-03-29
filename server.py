from flask import Flask, request, Response
import time
import cv2
import threading
import copy
import numpy as np

from utils import Algorithm, PoseEstimator, MonocularMapper


class ProcessingServer:

    def __init__(self):
        self.app = app = Flask(__name__)
        estimator = PoseEstimator('models/pose_landmarker_full.task')
        mapper = MonocularMapper(1)
        self.algoritmo = Algorithm(estimator, mapper)

    def processThisImage(self):
        r = request
        nparr = np.frombuffer(r.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        fall = self.algoritmo.run(img)

        cv2.putText(img, "FALL: "+str(fall), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 0, 255) if fall else (0, 255, 0)))

        _, img_encoded = cv2.imencode('.jpg', img)
        print(img_encoded)
        return Response(response=img_encoded.tobytes(), status=200, mimetype="image/jpeg")

    def startServer(self):
        self.app.add_url_rule('/processThisImage', "processThisImage", self.processThisImage, methods=['POST'])
        self.app.run(debug=True, host='0.0.0.0', use_reloader=False)


if __name__ == '__main__':
    server = ProcessingServer()
    server.startServer()