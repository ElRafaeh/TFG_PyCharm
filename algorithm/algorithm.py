import math
from time import perf_counter

from depthmap_extraction import MonocularMapper
from imager import Image
from pointcloud_extraction import PointCloud3D
from pose_extraction import PoseEstimator, Landmarks
import numpy as np


class Algorithm:
    def __init__(self, image: Image, estimator: PoseEstimator, mapper: MonocularMapper):
        self.image = image
        self.estimator = estimator
        self.mapper = mapper
        start = perf_counter()
        self.cloud = PointCloud3D.get_cloud_from_image(self.mapper, self.image.image)
        print(f'Elapsed cloud creation time: {perf_counter() - start:.3f}s')
        self.landmarks = estimator.get_landmarks(self.image.image)
        self.landmarks3D = None if not self.landmarks else self.cloud.get_landmarks_points(self.landmarks)

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

    def leg_distance(self):
        # d = ((x2 - x1)2 + (y2 - y1)2 + (z2 - z1)2)1/2
        if self.landmarks3D is None:
            print("No landmarks detected")
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

    def distance_to_plane(self):
        print(self.landmarks3D)
        # print
