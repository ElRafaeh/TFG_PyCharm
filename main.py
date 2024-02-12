import os
import argparse

from algorithm import Algorithm
from pointcloud_extraction import PointCloud3D
from depthmap_extraction import MonocularMapper
from imager import Image
from pose_extraction import PoseEstimator


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", "-i",
        type=str, required=True,
        help="The image file to get data from"
    )
    parser.add_argument(
        "--level", "-l",
        type=int, required=False, default=2,
        help="MiDaS model type"
    )
    parser.add_argument(
        '--model', '-m',
        type=str, required=True, default='',
        help='Model for pose estimation'
    )
    return parser


def main():
    args = get_parser().parse_args()
    image_path = os.path.join(os.getcwd(), args.image)
    model_path = os.path.join(os.getcwd(), args.model)

    image_sample = Image.from_file(image_path)
    estimator = PoseEstimator(model_path=model_path)
    mapper = MonocularMapper(args.level)

    algorithm = Algorithm(image_sample, estimator, mapper)
    print(algorithm.leg_distance())
    algorithm.show_landmarks_on_cloud(False)


if __name__ == "__main__":
    main()
