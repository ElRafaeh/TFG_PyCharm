from flask import Flask, request, Response
import cv2
import threading
import numpy as np
import telegram_send
import asyncio
from utils import Algorithm, MonocularMapper


class ProcessingServer:
  def __init__(self, app: Flask, algorithm):
    self.app = app
    self.algorithm = algorithm
    self.image = None
    self.fall = None
    self.thread = threading.Thread(target=self.show_image)
    self.count = 0
    self.running = True
    self.thread.start()
    
  def run(self, telegram_report = True):
    r = request    
    nparr = np.frombuffer(r.data, np.uint8)
    
    self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    self.fall = algoritmo.run(self.image)
    
    if telegram_report:
      self.report()
    
    return Response(response='RECEIVED', status=200, mimetype="text/xml")
  
  def report(self):
    if self.fall:
      self.count += 1
      if self.count == 10:
        _, img_encoded = cv2.imencode('.jpg', self.image)
        asyncio.run(telegram_send.send(captions=['⚠ ALERTA! Caida detectada!'], images=[img_encoded.tobytes()]))
      return
    self.count = 0
      
  
  def start(self):
    self.app.add_url_rule('/processThisImage', 'processThisImage', self.run, methods=['POST'])
    self.app.run(debug=True, host='0.0.0.0', use_reloader=False)
    self.running = False

  def show_image(self):
    while self.thread.is_alive() and self.running:
      if self.fall != None:
        reported_imge = self.image.copy()
        cv2.putText(reported_imge, "FALL: "+str(self.fall), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 0, 255) if self.fall else (0, 255, 0)))
        cv2.imshow('Detection', reported_imge)
        cv2.waitKey(1)
            
    print('Cerrando aplicación')
        
if __name__ == '__main__':
    algoritmo = Algorithm(mapper=MonocularMapper(1))
    app = Flask(__name__)
    
    server = ProcessingServer(app, algoritmo)
    server.start()