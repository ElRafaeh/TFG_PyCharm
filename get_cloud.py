import os
import argparse

from utils.depthmap_extraction import MonocularMapper
from utils.pointcloud_extraction import PointCloud3D
from utils.imager import Image


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
    return parser


def main():
    args = get_parser().parse_args()
    image_path = os.path.join(os.getcwd(), args.image)
    mapper = MonocularMapper(args.level)
    image_sample = Image.from_file(image_path)

    cloud = PointCloud3D.get_cloud_from_image(mapper, image_sample.image)
    cloud.draw_cloud()


if __name__ == "__main__":
    main()
