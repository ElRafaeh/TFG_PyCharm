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
    self.algorithm: Algorithm = algorithm
    self.landmarks = None
    self.image = None
    self.fall = None
    self.thread = threading.Thread(target=self.show_image)
    self.count = 0
    self.running = True
    self.thread.start()
    
  def run(self, telegram_report = True):
    r = request    
    nparr = np.frombuffer(r.data, np.uint8)
    
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    self.fall = self.algorithm.run(img)
    self.image = cv2.cvtColor(self.algorithm.image.image, cv2.COLOR_RGB2BGR)
    self.landmarks = self.algorithm.landmarks
    
    if telegram_report:
      self.report(img)
    
    return Response(response='RECEIVED', status=200, mimetype="text/xml")
  
  def report(self, img):
    if self.fall:
      self.count += 1
      if self.count == 10:
        _, img_encoded = cv2.imencode('.jpg', img)
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
        
        for i, landmark in enumerate(self.landmarks):
            cv2.circle(reported_imge, (int(landmark[0]), int(landmark[1])), 1, (255, 0, 0), 2)
            # cv2.scatter(int(landmark[0]), int(landmark[1]), c='#0CFF00', s=6)
            cv2.putText(reported_imge, str(i), (int(landmark[0])+1, int(landmark[1])+1), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)
        
        cv2.imshow('Detection', reported_imge)
        cv2.waitKey(1)
            
    print('Cerrando aplicación')
        
if __name__ == '__main__':
    algoritmo = Algorithm(3)
    app = Flask(__name__)
    
    server = ProcessingServer(app, algoritmo)
    server.start()