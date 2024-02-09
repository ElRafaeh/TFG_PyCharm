import os
import argparse
from pointcloud_extraction import PointCloud3D, MonocularMapper
from imager import Image
from pose_extraction import PoseEstimator
import open3d as o3d
import numpy as np
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

    # Convert the image to depth map
    mapper = MonocularMapper(args.level)
    raw_depth_map = mapper.map(image_sample.image)
    depth_map = Image.from_array(raw_depth_map)

    # Extraction of the point cloud from the depth map
    cloud = PointCloud3D(depth_map.image, image_sample.image / 255.0)
    # cloud.draw_cloud()

    # Estimation of the pose landmarks
    estimator = PoseEstimator(model_path=model_path)
    landmarks = estimator.get_landmarks(image_sample.image)
    # image_sample.show_landmarks(landmarks)

    # Draw the cloud with the landmarks
    cloud.draw_cloud_landmarks(landmarks, True)


if __name__ == "__main__":
    main()
