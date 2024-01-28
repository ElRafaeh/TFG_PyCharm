import argparse
from pose_extraction import PoseEstimator
from imager import Image
import os
import cv2


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m',
                        type=str, required=True, default='',
                        help='Model for pose estimation')
    parser.add_argument('--image', '-i',
                        type=str, required=True, default='',
                        help='Image path for pose estimation')
    return parser


def main():
    args = get_parser().parse_args()
    image_path = os.path.join(os.getcwd(), args.image)
    model_path = os.path.join(os.getcwd(), args.model)
    estimator = PoseEstimator(model_path=model_path)
    imager = Image.from_file(image_path)
    landmarks = estimator.get_landmarks(imager.image)
    imager.show_landmarks(landmarks)


if __name__ == "__main__":
    main()
