import cv2
from utils import PoseEstimator, Algorithm, MonocularMapper

photo = cv2.imread('images/mayor.png')
estimator = PoseEstimator('models/pose_landmarker_lite.task')
mapper = MonocularMapper(1)
algoritmo = Algorithm(estimator, mapper)
algoritmo.run(photo, debug=True)

