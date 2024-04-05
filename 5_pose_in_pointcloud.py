from utils import PoseEstimator, Image, PointCloud3D, MonocularMapper

img = Image.from_file('images/mayor.png')
estimator = PoseEstimator('models/pose_landmarker_full.task')
landmarks = estimator.get_landmarks(img.image)

cloud = PointCloud3D.get_cloud_from_image(MonocularMapper(1), img.image)

cloud.draw_cloud_landmarks3d(cloud.get_landmarks_points(landmarks), False)
