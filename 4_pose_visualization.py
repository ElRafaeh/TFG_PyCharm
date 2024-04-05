from utils import PoseEstimator, Image
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
import mediapipe as mp
import cv2

img = Image.from_file('images/mayor.png')
estimator = PoseEstimator('models/pose_landmarker_full.task')

landmarks = estimator.get_landmarks(img.image)

img.show_landmarks(landmarks)

# mp_image = mp.Image(mp.ImageFormat.SRGB, data=img.image)
# detections = estimator.detector.detect(mp_image)

# landmarks = detections.pose_landmarks[0]

# annotated_image = np.copy(img.image)

# # Draw the pose landmarks.
# pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
# pose_landmarks_proto.landmark.extend([
#   landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
# ])
# solutions.drawing_utils.draw_landmarks(
#   annotated_image,
#   pose_landmarks_proto,
#   solutions.pose.POSE_CONNECTIONS,
#   solutions.drawing_styles.get_default_pose_landmarks_style())

# cv2.imshow('a', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)