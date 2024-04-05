from utils import MonocularMapper, PointCloud3D, Image


img = Image.from_file('images/mayor.png')

cloud = PointCloud3D.get_cloud_from_image(MonocularMapper(1), img.image)
cloud.draw_cloud()