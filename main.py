import os
import argparse

from utils import Algorithm
from utils import MonocularMapper
from utils import Image
from utils import PoseEstimator


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
    # algorithm.show_landmarks_on_cloud(True)
    leg_distance = algorithm.leg_distance()
    distances = algorithm.distance_landmarks_to_plane()

    print('Has fallen?: ', Algorithm.detect_fall(leg_distance, distances))


if __name__ == "__main__":
    main()
