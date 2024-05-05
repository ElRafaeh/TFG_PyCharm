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
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, l_frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                cap.release()
                break
            self.frame = copy.copy(l_frame)
            # print(frame)
        self.running = False

    def startCamera(self):
        self.running = True
        self.thread = threading.Thread(target=self.threadImage)
        self.thread.start()
        print("Camera streaming now")

    def start(self):
        self.startCamera()

        headers = {'contest-type': 'image/jpg'}
        pp = pprint.PrettyPrinter(indent=4)

        while self.running:
            if self.frame is None:
                continue

            _, img_encoded = cv2.imencode('.jpg', self.frame.copy())
            response = requests.post(self.url_server, data=img_encoded.tobytes(), headers=headers) 
            print(response.text)
            
        # response = requests.post(self.url_server, data='CLOSED', headers={'contest-type': 'text/xml'}) 
        # print(response.text)

if __name__ == '__main__':
    addr = 'http://localhost:5000'
    test_url = addr + '/processThisImage'

    server = CameraClient(test_url)
    server.start()