from utils import Algorithm
import cv2
from flask import Flask, request, Response
import threading
import numpy as np
import telegram_send
import asyncio

algoritmo = Algorithm(2)  
  
class FallDetector:
  def __init__(self, camera: cv2.VideoCapture | Flask, server, telegram_report=True):
    self.camera = camera
    self.telegram_report = telegram_report
    self.server = server
    
    self.fall_count = 0
    self.no_fall_count = 0
    self.running = True
    self.image = None
    self.fall = None
    self.fall_aux = None
    
  def run_camera(self):
    while self.camera.isOpened():
      ret, frame = self.camera.read()
      
      if not ret:
        break
      
      self.image = frame
      self.fall_aux = algoritmo.run(self.image)
      
      # print(self.fall)
      
      if self.telegram_report:
        self.fall = self.report()
    
  
  def show_image(self):
    while self.running:
      if self.fall != None:
        reported_imge = self.image.copy()
        cv2.putText(reported_imge, "FALL: "+str(self.fall), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 0, 255) if self.fall else (0, 255, 0)))
        cv2.imshow('Detection', reported_imge)
        cv2.waitKey(1)
            
    print('Cerrando aplicación')
    
  def report(self):
    filter = 10
    print(self.fall_count, self.no_fall_count)
    
    if self.fall_aux:
      self.fall_count += 1
      self.no_fall_count = 0
      
      if self.fall_count == filter:
        _, img_encoded = cv2.imencode('.jpg', self.image)
        asyncio.run(telegram_send.send(captions=['⚠ ALERTA! Caida detectada!'], images=[img_encoded.tobytes()]))
        return True
      elif self.fall_count > filter:
        return True
      return False 
    else:
      self.no_fall_count += 1
      self.fall_count = 0
      if self.no_fall_count >= filter:
        return False
      return True
    
    
  def detection_server_image(self):
    r = request    
    nparr = np.frombuffer(r.data, np.uint8)
    
    self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    self.fall = algoritmo.run(self.image)
    
    if self.telegram_report:
      self.report()
    
    return Response(response='RECEIVED', status=200, mimetype="text/xml")

  def run_server(self):
    self.camera.add_url_rule('/processThisImage', 'processThisImage', self.detection_server_image, methods=['POST'])
    self.camera.run(debug=True, host='0.0.0.0', use_reloader=False)
    
  def start(self):
    thread = threading.Thread(target=self.show_image)
    thread.start()
    
    if self.server:
      self.run_server()
    else:
      self.run_camera()
    self.running = False


        
if __name__ == '__main__':
  camera = cv2.VideoCapture('datasets/imvia/Home_02/Videos/video (39).avi') 
  fall_detector = FallDetector(camera, False, True)
  fall_detector.start()




# class ProcessingServer:
#   def __init__(self, app: Flask, algorithm):
#     self.app = app
#     self.algorithm = algorithm
#     self.image = None
#     self.fall = None
#     self.thread = threading.Thread(target=self.show_image)
#     self.count = 0
#     self.running = True
#     self.thread.start()
    
#   def run(self, telegram_report = True):
#     r = request    
#     nparr = np.frombuffer(r.data, np.uint8)
    
#     self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     self.fall = algoritmo.run(self.image)
    
#     if telegram_report:
#       self.report()
    
#     return Response(response='RECEIVED', status=200, mimetype="text/xml")
  
#   def report(self):
#     if self.fall:
#       self.count += 1
#       if self.count == 10:
#         _, img_encoded = cv2.imencode('.jpg', self.image)
#         asyncio.run(telegram_send.send(captions=['⚠ ALERTA! Caida detectada!'], images=[img_encoded.tobytes()]))
#       return
#     self.count = 0
      
  
#   def start(self):
#     self.app.add_url_rule('/processThisImage', 'processThisImage', self.run, methods=['POST'])
#     self.app.run(debug=True, host='0.0.0.0', use_reloader=False)
#     self.running = False

#   def show_image(self):
#     while self.thread.is_alive() and self.running:
#       if self.fall != None:
#         reported_imge = self.image.copy()
#         cv2.putText(reported_imge, "FALL: "+str(self.fall), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 0, 255) if self.fall else (0, 255, 0)))
#         cv2.imshow('Detection', reported_imge)
#         cv2.waitKey(1)
            
#     print('Cerrando aplicación')