import os
import argparse
from pointcloud_extraction import PointCloud, MonocularMapper
from imager import Image
from matplotlib import pyplot as plt


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
    raw_depth_map = mapper.map(image_sample.image)
    depth_map = Image.from_array(raw_depth_map)
    cloud = PointCloud(depth_map.image)
    cloud.draw_cloud()
    cloud.save()

if __name__ == "__main__":
    main()
