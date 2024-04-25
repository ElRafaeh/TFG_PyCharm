from utils import MonocularMapper, PointCloud3D, Image
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

img = Image.from_file('images/prueba_rafa.png')
mapper = MonocularMapper(3)

cloud = PointCloud3D.get_cloud_from_image(mapper, cv2.resize(img.image, (640, 360)))

# cloud.draw_cloud()
insiders, plane_models = cloud.get_multiple_planes()
# insider_cloud, plane_model = cloud.get_segmented_cloud()

print(plane_models)
print(plane_models[:, 2])
index = np.argmin(plane_models[:, 2])


vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(insiders[index])

vis2 = o3d.visualization.Visualizer()
vis2.create_window()

for insider in insiders:
  vis2.add_geometry(insider)

while True:
  vis.poll_events()
  vis2.poll_events()
  