import requests
import time
import cv2
import threading
import copy
import numpy as np
import pprint


class CameraClient:

    def __init__(self, url):

        self.frame = None
        self.running = False
        self.thread = None
        self.url_server = url

    def threadImage(self):
        cap = cv2.VideoCapture(-1)
        while self.running:
            ret, l_frame = cap.read()
            self.frame = copy.copy(l_frame)
            # print(frame)

    def startCamera(self):
        self.running = True
        self.thread = threading.Thread(target=self.threadImage)
        self.thread.start()
        print("Camera streaming now")

    def start(self):
        self.startCamera()

        headers = {'contest-type': 'image/jpg'}
        pp = pprint.PrettyPrinter(indent=4)

        while True:
            if self.frame is None:
                continue

            cv2.imshow("w enviado", self.frame)

            _, img_encoded = cv2.imencode('.jpg', self.frame.copy())
            response = requests.post(self.url_server, data=img_encoded.tobytes(), headers=headers)  # .raw
            nparr = np.frombuffer(response._content, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            cv2.imshow("w recibido", img_np)
            cv2.waitKey(10)


if __name__ == '__main__':
    addr = 'http://localhost:5000'
    test_url = addr + '/processThisImage'

    server = CameraClient(test_url)

    server.start()