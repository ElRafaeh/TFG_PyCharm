import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseEstimator(object):
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def get_landmarks(self, image, image_format=mp.ImageFormat.SRGB):
        mp_image = mp.Image(image_format, data=image)
        detections = self.detector.detect(mp_image)

        if not detections.pose_landmarks:
            return []

        landmarks = detections.pose_landmarks[0]

        # detector = [
        #
        # ]
        return [
            (int(landmark.x*image.shape[1]), int(landmark.y*image.shape[0]))
            for landmark in landmarks
            if int(landmark.x * image.shape[1]) < image.shape[1] and int(landmark.y * image.shape[0]) < image.shape[0]
        ]










