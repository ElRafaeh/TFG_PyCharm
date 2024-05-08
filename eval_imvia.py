import numpy as np
import cv2
from utils import Algorithm, MonocularMapper
from time import perf_counter

mapper = 1

class Evaluador:
  def __init__(self, path, dataset, directory, first, last) -> None:
    self.path = f'{path}/{dataset}/{directory}'
    self.first = first
    self.last = last
    self.algorithm = Algorithm(mapper=MonocularMapper(mapper))
    
  def run_algorithm(self, index):
    video = cv2.VideoCapture(f'{self.path}/Videos/video ({index}).avi')
    falls = []
    
    print(f'Estimating falls in the video {index}')
    
    while video.isOpened():
      ret, frame = video.read()
      
      if not ret:
          # print("Can't receive frame (stream end?). Exiting ...")
          break

      fall = self.algorithm.run(frame)

      # cv2.putText(frame, "FALL: "+str(fall), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
      # cv2.putText(frame, "FALL: "+str(fall), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
      # cv2.imshow('frame', frame)
      
      falls.append(fall)
      
      # if cv2.waitKey(1) == ord('q'):
      #     break
          
    video.release()
     # When everything done, release the capture
    # cv2.destroyAllWindows()  
    
    # print('Estimatión finished.')
    
    return np.asarray(falls)  
  
  def read_annotations(self, index):
    with open(f'{self.path}/Annotation_files/video ({index}).txt', 'r') as file:
      start_fall = int(file.readline()) - 1 # LOS FRAMES EMPIEZAN POR 1, ENTONCES RESTAMOS UNO PARA QUE EMPIEZEN DE 0
      end_fall = int(file.readline())
      frames = len(file.readlines()) # LOS FICHEROS CONTIENEN UN INTRO FINAL VACÍO, POR ESO -1
    falses = np.full(frames, False, dtype=bool)
    falses[start_fall:end_fall] = True
    
    return falses
          
  
  def eval(self):
    with open(f'results/{self.path}/results_{mapper}.txt', 'w') as file:
      for index in range(self.first, self.last+1):
        real_falls = self.read_annotations(index)
        estimated_falls = self.run_algorithm(index)
        compared_array = real_falls == estimated_falls
        file.write(f'Video {index} & {round((compared_array.sum() / len(real_falls)) * 100, 2)}\\% \\\\ \n')
      file.close()




if __name__ == '__main__':
  path = 'datasets'
  dataset = 'imvia'
  directory = 'Coffee_room_01'
  eval = Evaluador(path, dataset, directory, first=1, last=48)
  start = perf_counter()
  eval.eval()
  end = perf_counter()
  with open(f'results/{eval.path}/time_results_{mapper}.txt', 'w') as file:
    file.write(f'Elapsed time: {end - start} seconds')
    file.close()