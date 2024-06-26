import math
from time import perf_counter

from utils.depthmap_extraction import MonocularMapper
from utils.imager import Image
from utils.pointcloud_extraction import PointCloud3D
from utils.pose_extraction import PoseEstimator, Landmarks
import numpy as np
import cv2

factor = 0.7

class Algorithm:
    def __init__(self, mapper=1):
        self.estimator = PoseEstimator('models/pose_landmarker_full.task')
        self.mapper = MonocularMapper(mapper)
        self.image = None
        self.cloud = None
        self.landmarks = []
        self.landmarks3D = []
        self.fall_counter = 0
        self.no_fall_counter = 10
        self.prev_fall = None
        
        # start = perf_counter()
        # self.cloud = PointCloud3D.get_cloud_from_image(self.mapper, self.image.image)
        # print(f'Elapsed cloud creation time: {perf_counter() - start:.3f}s')
        # self.landmarks = estimator.get_landmarks(self.image.image)
        # self.landmarks3D = None if not self.landmarks else self.cloud.get_landmarks_points(self.landmarks)

    def show_landmarks_on_image(self):
        if not len(self.landmarks):
            print("No landmarks detected")
            return
        self.image.show_landmarks(self.landmarks)

    def show_landmarks_on_cloud(self, plane=False):
        if not len(self.landmarks3D):
            print("No landmarks detected")
            return
        self.cloud.draw_cloud_landmarks3d(self.landmarks3D, plane)

    def detect_fall(self):
        leg_distance = self.leg_distance()
        if leg_distance is None or self.prev_fall is None:
            return False

        distances = self.distance_landmarks_to_plane()
        
        # or \
            # (np.all(distances[11:22:2] < leg_distance) and np.all(distances[23:32:2] < leg_distance)) or \
            # (np.all(distances[12:23:2] < leg_distance) and np.all(distances[24:32:2] < leg_distance))
        
        if (np.all(distances[23:25] < leg_distance) and np.all(distances[11:13] < leg_distance)) or \
            (np.all(distances[11:22:2] < leg_distance) and np.all(distances[23:32:2] < leg_distance)) or \
            (np.all(distances[12:23:2] < leg_distance) and np.all(distances[24:32:2] < leg_distance)):
            return True   

        return False
    
    def filter_ouliers(self):
        if self.prev_fall:
            self.fall_counter += 1
            if self.fall_counter > 5:
                self.no_fall_counter = 0
                return True
            return False
        else:
            self.no_fall_counter += 1
            if self.no_fall_counter > 5:
                self.fall_counter = 0
                return False  
            return True                     

    def run(self, image, debug=False):
        # self.image = Image.from_array(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 360), interpolation=cv2.INTER_LINEAR))
        self.image = Image.from_array(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (320, 240), interpolation=cv2.INTER_CUBIC))
        # self.image = Image.from_array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        start = perf_counter()
        self.cloud = PointCloud3D.get_cloud_from_image(self.mapper, self.image.image, debug)
        start2 = perf_counter()
        self.landmarks = self.estimator.get_landmarks(self.image.image)
        self.landmarks3D = [] if not self.landmarks else self.cloud.get_landmarks_points(self.landmarks)
        end2 = perf_counter()
        
        if debug:
            print(f'Elapsed landmarks time: {end2 - start2:.3f}s') 
            print(f'Elapsed algorithm time: {end2 - start:.3f}s')
            # self.show_landmarks_on_image()
            # self.show_landmarks_on_cloud()
            self.cloud.draw_cloud()
            self.show_landmarks_on_cloud(True)
            
        self.prev_fall = self.detect_fall()
        # print(self.prev_fall)
        return self.filter_ouliers()
        # print('FALL:' + str(self.detect_fall()), end='\r')

    def leg_distance(self):
        # d = ((x2 - x1)2 + (y2 - y1)2 + (z2 - z1)2)1/2
        if not len(self.landmarks3D) or len(self.landmarks) < 25:
            return 
        
        if len(self.landmarks) == 33:
            return max(
                math.fabs(
                    math.sqrt(
                        (self.landmarks3D[Landmarks['RIGHT_ANKLE']][0] - self.landmarks3D[Landmarks['RIGHT_KNEE']][0])**2 +
                        (self.landmarks3D[Landmarks['RIGHT_ANKLE']][1] - self.landmarks3D[Landmarks['RIGHT_KNEE']][1])**2 +
                        (self.landmarks3D[Landmarks['RIGHT_ANKLE']][2] - self.landmarks3D[Landmarks['RIGHT_KNEE']][2])**2
                    )
                ),
                math.fabs(
                    math.sqrt(
                        (self.landmarks3D[Landmarks['LEFT_ANKLE']][0] - self.landmarks3D[Landmarks['LEFT_KNEE']][0])**2 +
                        (self.landmarks3D[Landmarks['LEFT_ANKLE']][1] - self.landmarks3D[Landmarks['LEFT_KNEE']][1])**2 +
                        (self.landmarks3D[Landmarks['LEFT_ANKLE']][2] - self.landmarks3D[Landmarks['LEFT_KNEE']][2])**2
                    )
                )
            )*factor
        else:
            return max(
                math.fabs(
                    math.sqrt(
                        (self.landmarks3D[Landmarks['LEFT_ELBOW']][0] - self.landmarks3D[Landmarks['LEFT_INDEX']][0])**2 +
                        (self.landmarks3D[Landmarks['LEFT_ELBOW']][1] - self.landmarks3D[Landmarks['LEFT_INDEX']][1])**2 +
                        (self.landmarks3D[Landmarks['LEFT_ELBOW']][2] - self.landmarks3D[Landmarks['LEFT_INDEX']][2])**2
                    )
                ),
                math.fabs(
                    math.sqrt(
                        (self.landmarks3D[Landmarks['RIGHT_ELBOW']][0] - self.landmarks3D[Landmarks['RIGHT_INDEX']][0])**2 +
                        (self.landmarks3D[Landmarks['RIGHT_ELBOW']][1] - self.landmarks3D[Landmarks['RIGHT_INDEX']][1])**2 +
                        (self.landmarks3D[Landmarks['RIGHT_ELBOW']][2] - self.landmarks3D[Landmarks['RIGHT_INDEX']][2])**2
                    )
                )
            )*factor

    def distance_landmarks_to_plane(self):
        if not len(self.landmarks3D):
            return []

        return (
            np.abs(
                self.cloud.plane[0] * self.landmarks3D[:, 0] +
                self.cloud.plane[1] * self.landmarks3D[:, 1] +
                self.cloud.plane[2] * self.landmarks3D[:, 2] +
                self.cloud.plane[3] * np.ones(self.landmarks3D.shape[0])
            ) / np.sqrt(self.cloud.plane[0]**2 + self.cloud.plane[1]**2 + self.cloud.plane[2]**2)
        )
