import requests
import cv2
import copy
import pprint


class CameraClient:
    def __init__(self, url):
        self.frame = None
        self.running = False
        self.thread = None
        self.url_server = url

    def start(self):
        cap = cv2.VideoCapture('datasets/imvia/Coffee_room_01/Videos/video (11).avi')

        headers = {'contest-type': 'image/jpg'}
        pp = pprint.PrettyPrinter(indent=4)

        while cap.isOpened():
            ret, l_frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                cap.release()
                break
            
            self.frame = copy.copy(l_frame)
            _, img_encoded = cv2.imencode('.jpg', self.frame)
            response = requests.post(self.url_server, data=img_encoded.tobytes(), headers=headers) 
            print(response.text)
            
        # response = requests.post(self.url_server, data='CLOSED', headers={'contest-type': 'text/xml'}) 
        print(response.text)

if __name__ == '__main__':
    addr = 'http://localhost:5000'
    test_url = addr + '/processThisImage'

    server = CameraClient(test_url)
    server.start()