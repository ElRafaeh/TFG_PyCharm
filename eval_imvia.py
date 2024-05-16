import numpy as np
import cv2
from utils import Algorithm, MonocularMapper
from time import perf_counter
import os
import sys


def confusion_matrix(y_actual, y_pred):
  TP = 0
  FP = 0
  TN = 0
  FN = 0

  for i in range(len(y_pred)): 
      if y_actual[i]==y_pred[i]==1:
          TP += 1
      if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
          FP += 1
      if y_actual[i]==y_pred[i]==0:
          TN += 1
      if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
          FN += 1

  return TP, FP, TN, FN    

class Evaluador:
  def __init__(self, path, dataset, directory, first, last) -> None:
    self.path = f'{path}/{dataset}/{directory}'
    self.first = first
    self.last = last
    self.algorithm = Algorithm(mapper=MonocularMapper(1))
    
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
      
          
    video.release()
     # When everything done, release the capture
    # cv2.destroyAllWindows()  
        
    return np.asarray(falls)  
  
  def read_annotations(self, index):
    with open(f'{self.path}/Annotation_files/video ({index}).txt', 'r') as file:
      start_fall = int(file.readline()) - 1 # LOS FRAMES EMPIEZAN POR 1, ENTONCES RESTAMOS UNO PARA QUE EMPIEZEN DE 0
      end_fall = int(file.readline())
      
    video = cv2.VideoCapture(f'{self.path}/Videos/video ({index}).avi')
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    
    falses = np.full(frames, False, dtype=bool)
    falses[start_fall:end_fall] = True
    
    return falses
  

  
  def eval(self):
    with open(f'results/{self.path}/results.txt', 'w') as file:
      for index in range(self.first, self.last+1):
        # with open(f'results/{self.path}/videos/video_{index}.txt', 'w') as arr_file:
          real_falls = self.read_annotations(index)
          start = perf_counter()
          estimated_falls = self.run_algorithm(index)
          end = perf_counter()
          TP, FP, TN, FN = confusion_matrix(real_falls, estimated_falls)
          compared_array = real_falls == estimated_falls
          
          file.write(f'Video {index} & {round((compared_array.sum() / len(real_falls)) * 100, 2)}\\% & {round((end - start)/len(estimated_falls), 2)} & {TP} & {FP} & {TN} & {FN} \\\\ \n')
          # arr_file.write(f'{TP}\n{FP}\n{TN}\n{FN}\n')
      file.close()




if __name__ == '__main__':
  path = 'datasets'
  dataset = 'imvia'  
  np.set_printoptions(threshold=sys.maxsize)
  
  
  for dir in os.listdir(f'{path}/{dataset}'):
    print('#########################################################################')
    print(f'##########{dir}###########')
    print('#########################################################################')
    f = open(f'{path}/{dataset}/{dir}/data.txt', 'r')
    eval = Evaluador(path, dataset, dir, first=int(f.readline()), last=int(f.readline()))
    eval.eval()
    f.close()