import requests
import time
import cv2
import threading
import copy
import numpy as np
import pprint


if __name__ == '__main__':
  addr = 'http://127.0.0.1:36797'
  test_url = addr + '/processThisImage'
  
  frame = cv2.imread('images/mayor.png')
  headers = {'contest-type': 'image/jpg'}
  

  _, img_encoded = cv2.imencode('.jpg', frame)
  response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers)  # .raw
  nparr = np.frombuffer(response._content, np.uint8)
  img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

  cv2.imshow("w recibido", img_np)
  cv2.waitKey(0)