from utils import MonocularMapper, PointCloud3D, Image
import cv2


img = Image.from_file('images/prueba_rafa.png')

cloud = PointCloud3D.get_cloud_from_image(
                                          MonocularMapper(2), 
                                          cv2.resize(img.image, (640, 360)), 
                                          True
                                        )

cloud.draw_cloud()
# cl, ind = cloud.cloud.remove_statistical_outlier(nb_neighbors=20,
#                                                     std_ratio=2.0)

# inlier_cloud = cloud.cloud.select_by_index(ind)
# outlier_cloud = cloud.cloud.select_by_index(ind, invert=True)

# print("Showing outliers (red) and inliers (gray): ")
# outlier_cloud.paint_uniform_color([1, 0, 0])
# # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
# o3d.visualization.draw_geometries([inlier_cloud],
#                               zoom=0.3412,
#                               front=[0.4257, -0.2125, -0.8795],
#                               lookat=[2.6172, 2.0475, 1.532],
#                               up=[-0.0694, -0.9768, 0.2024])
