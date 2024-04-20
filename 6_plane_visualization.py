from utils import MonocularMapper, PointCloud3D, Image


img = Image.from_file('images/prueba_rafa.png')

cloud = PointCloud3D.get_cloud_from_image(MonocularMapper(3), img.image, True)
# cloud.get_segmented_cloud()
# cloud.draw_cloud()