import math
from time import perf_counter

from utils.depthmap_extraction import MonocularMapper
from utils.imager import Image
from utils.pointcloud_extraction import PointCloud3D
from utils.pose_extraction import PoseEstimator, Landmarks
import numpy as np
import cv2


class Algorithm:
    def __init__(self, estimator: PoseEstimator, mapper: MonocularMapper):
        self.estimator = estimator
        self.mapper = mapper
        self.image = None
        self.cloud = None
        self.landmarks = None
        self.landmarks3D = None
        # start = perf_counter()
        # self.cloud = PointCloud3D.get_cloud_from_image(self.mapper, self.image.image)
        # print(f'Elapsed cloud creation time: {perf_counter() - start:.3f}s')
        # self.landmarks = estimator.get_landmarks(self.image.image)
        # self.landmarks3D = None if not self.landmarks else self.cloud.get_landmarks_points(self.landmarks)

    def show_landmarks_on_image(self):
        if self.landmarks is None:
            print("No landmarks detected")
            return
        self.image.show_landmarks(self.landmarks)

    def show_landmarks_on_cloud(self, plane=False):
        if self.landmarks3D is None:
            print("No landmarks detected")
            return
        self.cloud.draw_cloud_landmarks3d(self.landmarks3D, plane)

    def run(self, image, debug=False):
        start = perf_counter()
        self.image = Image.from_array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.cloud = PointCloud3D.get_cloud_from_image(self.mapper, self.image.image)
        self.landmarks = self.estimator.get_landmarks(self.image.image)
        self.landmarks3D = None if not self.landmarks else self.cloud.get_landmarks_points(self.landmarks)
        
        if debug:
            print(f'Elapsed cloud creation time: {perf_counter() - start:.3f}s')
            self.show_landmarks_on_image()
            self.show_landmarks_on_cloud()
            self.show_landmarks_on_cloud(True)

        return self.detect_fall()
        # print('FALL:' + str(self.detect_fall()), end='\r')

    def leg_distance(self):
        # d = ((x2 - x1)2 + (y2 - y1)2 + (z2 - z1)2)1/2
        if self.landmarks3D is None or len(self.landmarks) < 33:
            # print("No landmarks detected")
            return

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
        )

    def distance_landmarks_to_plane(self):
        if self.landmarks3D is None:
            return

        return (
            np.abs(
                self.cloud.plane[0] * self.landmarks3D[:, 0] +
                self.cloud.plane[1] * self.landmarks3D[:, 1] +
                self.cloud.plane[2] * self.landmarks3D[:, 2] +
                self.cloud.plane[3] * np.ones(self.landmarks3D.shape[0])
            ) / np.sqrt(self.cloud.plane[0]**2 + self.cloud.plane[1]**2 + self.cloud.plane[2]**2)
        )

    def detect_fall(self):
        leg_distance = self.leg_distance()
        if leg_distance is None:
            return False

        distances = self.distance_landmarks_to_plane()
        if np.all(distances[23:] < leg_distance):
            return True
        elif np.all(distances[:23] < leg_distance):
            return True

        return False
