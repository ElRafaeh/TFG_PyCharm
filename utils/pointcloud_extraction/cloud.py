import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud
from utils.imager import Image


class PointCloud3D(object):
    def __init__(self, array: np.ndarray, colored_array: np.ndarray):
        """
            Extract the PointCloud of a depth map image

            Args:
                array: DepthMap image to extract pointcloud
                colored_array: RGB image for extract colors to pointcloud, values [0...1]
        """
        self.scale_factor = self.avg(array.shape) / array.max()
        self.image = array
        self.array = self.__preprocess(array, self.scale_factor)
        self.colors = colored_array.reshape((-1, colored_array.shape[-1]))
        self.cloud = self.__create_pcd(self.array, self.colors)
        self.segmented_cloud, self.plane = self.get_segmented_cloud()

    @staticmethod
    def avg(d):
        return sum(d) / len(d)

    @staticmethod
    def __preprocess(array, scale_factor):
        height, width = array.shape

        mod_depth = array * scale_factor
        xyz = np.empty((height, width, 3))
        xyz[:, :, 0] = np.arange(height)[:, np.newaxis]
        xyz[:, :, 1] = np.arange(width)
        xyz[:, :, 2] = mod_depth

        print(f'Converted raw array of shape {array.shape} to RGB array of shape {(height*width, 3)}')
        return xyz.reshape((-1, 3))
    
    @staticmethod
    def __create_pcd(array, colors) -> PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(array)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def draw_cloud(self) -> None:
        # Visualize point cloud from array
        o3d.visualization.draw_geometries([self.cloud])

    def draw_cloud_landmarks3d(self, landmarks, plane_cloud=True) -> None:
        landmarks_pcd = self.__create_pcd(array=landmarks,
                                          colors=np.tile([1.0, 0, 0], (landmarks.shape[0], 1)))
        o3d.visualization.draw_geometries([landmarks_pcd, self.segmented_cloud if plane_cloud else self.cloud])

    def draw_cloud_landmarks(self, landmarks, plane_cloud=True) -> None:
        landmarks_points = self.get_landmarks_points(landmarks)
        landmarks_pcd = self.__create_pcd(array=landmarks_points,
                                          colors=np.tile([1.0, 0, 0], (landmarks_points.shape[0], 1)))
        o3d.visualization.draw_geometries([landmarks_pcd, self.segmented_cloud if plane_cloud else self.cloud])

    def draw_voxels(self):
        try: 
            n = 1000
            pcd = self.cloud
            pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
            pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(n, 3)))
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=0.01)
            o3d.visualization.draw_geometries([voxel_grid])
            print(f'Displaying 3D data for {voxel_grid:,} data points')  

        except Exception as e:
            print(f'Failed to draw voxel grid: {e}')

    @classmethod
    def get_cloud_from_image(cls, mapper, image):
        raw_depth_map = mapper.map(image)
        depth_map = Image.from_array(raw_depth_map)

        return cls(depth_map.image, image / 255.0)

    def get_segmented_cloud(self) -> tuple[PointCloud, np.ndarray]:
        plane_model, insiders = self.cloud.segment_plane(distance_threshold=10,
                                                         ransac_n=3,
                                                         num_iterations=1000)

        # [a, b, c, d] = plane_model
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        insider_cloud: o3d.geometry.PointCloud = self.cloud.select_by_index(insiders)
        insider_cloud.colors = o3d.utility.Vector3dVector(self.colors[insiders])

        return insider_cloud, plane_model

    def get_landmarks_points(self, landmarks):
        landmarks_points = []

        for landmark in landmarks:
            mod_depth = self.image[landmark[1]][landmark[0]] * self.scale_factor
            xyz = np.array([landmark[1], landmark[0], mod_depth])
            landmarks_points.append(xyz)

        return np.array(landmarks_points)

    def save(self, filename):
        o3d.io.write_point_cloud(f'{filename}.pcd', self.cloud)
    