from flask import Flask, request, Response
import cv2
import threading
import numpy as np
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

from utils import Algorithm, PoseEstimator, MonocularMapper


class ProcessingServer:
  def __init__(self, app: Flask, algorithm):
    self.app = app
    self.algorithm = algorithm
    self.image = None
    self.fall = None
    self.thread = threading.Thread(target=self.show_image)
    self.running = True
    self.thread.start()
    
  def run(self):
    r = request    
    nparr = np.frombuffer(r.data, np.uint8)
    self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    start_time = perf_counter()
    # self.fall = algoritmo.run(self.image)
    end_time = perf_counter()
    
    # print(f'Elapsed algorithm time: {end_time - start_time:.3f}s')

    # print('LLEGO')

    return Response(response='RECEIVED', status=200, mimetype="text/xml")
  
  def start(self):
    self.app.add_url_rule('/processThisImage', 'processThisImage', self.run, methods=['POST'])
    self.app.run(debug=True, host='0.0.0.0', use_reloader=False)
    self.running = False

  def show_image(self):
    while self.thread.is_alive() and self.running:
      if self.fall != None:
        # cv2.putText(self.image, "FALL: "+str(self.fall), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 0, 255) if self.fall else (0, 255, 0)))
        cv2.imshow('Detection', self.image)
        if cv2.waitKey(1) == ord('q'):
          raise(KeyboardInterrupt)
          break
            
    print('Cerrando aplicación')
        
if __name__ == '__main__':
    estimator = PoseEstimator('models/pose_landmarker_full.task')
    mapper = MonocularMapper(1)
    algoritmo = Algorithm(estimator, mapper)
    app = Flask(__name__)
    
    server = ProcessingServer(app, algoritmo)
    server.start()