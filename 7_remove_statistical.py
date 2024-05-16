from utils import MonocularMapper, PointCloud3D, Image
import cv2
import open3d as o3d

img = Image.from_file('images/mayor.png')

cloud = PointCloud3D.get_cloud_from_image(
                                          MonocularMapper(1), 
                                          cv2.resize(img.image, (640, 360)), 
                                        )

cl, ind = cloud.cloud.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
inlier_cloud = cloud.cloud.select_by_index(ind)
## DESCOMMENT FOR VISUALIZATION
# print("Showing outliers (red) and inliers (gray): ")
outlier_cloud = cloud.cloud.select_by_index(ind, invert=True)
outlier_cloud.paint_uniform_color([1, 0, 0])
# inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
o3d.visualization.draw_geometries([inlier_cloud])

