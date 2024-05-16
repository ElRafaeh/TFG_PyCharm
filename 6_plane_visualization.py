from utils import MonocularMapper, PointCloud3D, Image
import open3d as o3d


img = Image.from_file('images/mayor.png')

cloud = PointCloud3D.get_cloud_from_image(MonocularMapper(1), img.image, True)
o3d.visualization.draw_geometries([cloud.segmented_cloud])

# cloud.draw_cloud()