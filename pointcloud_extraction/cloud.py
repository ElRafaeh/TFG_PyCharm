import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud


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
        self.array, self.colors = self.__preprocess(array, self.scale_factor, colored_array)
        self.cloud = self.__create_pcd(self.array, self.colors)
        self.plane = self.segmented_cloud()

    @staticmethod
    def avg(d):
        return sum(d) / len(d)

    @staticmethod
    def __preprocess(array, scale_factor, colors):
        rgb = []
        point_colors = []
        height, width = array.shape

        for row in range(height):
            for col in range(width):
                mod_depth = array[row][col]*scale_factor
                xyz = np.array([row, col, mod_depth])
                rgb.append(xyz)
                point_colors.append(colors[row][col])

        rgb = np.array(rgb)
        print(f'Converted raw array of shape {array.shape} to RGB array of shape {rgb.shape}')
        return rgb, np.array(point_colors)
    
    @staticmethod
    def __create_pcd(array, colors) -> PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(array)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def draw_cloud(self):
        count, dim = self.array.shape
        try:
            if dim != 3:
                raise ValueError(f'Expected 3 dimensions but got {dim}')

            # Visualize point cloud from array
            print(f'Displaying 3D data for {count:,} data points')
            o3d.visualization.draw_geometries([self.cloud])
        except Exception as e:
            print(f'Failed to draw point cloud: {e}')

    def segmented_cloud(self):
        plane_model, inliers = self.cloud.segment_plane(distance_threshold=15,
                                                        ransac_n=3,
                                                        num_iterations=1000)

        # [a, b, c, d] = plane_model
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud: o3d.geometry.PointCloud = self.cloud.select_by_index(inliers)
        inlier_cloud.colors = o3d.utility.Vector3dVector(self.colors[inliers])

        return inlier_cloud

    def draw_cloud_landmarks(self, landmarks, plane_cloud=True):
        count, dim = self.array.shape
        try: 
            if dim != 3:
                raise ValueError(f'Expected 3 dimensions but got {dim}')
            landmarks_points = []

            for landmark in landmarks:
                mod_depth = self.image[landmark[1]][landmark[0]] * self.scale_factor
                xyz = np.array([landmark[1], landmark[0], mod_depth])
                landmarks_points.append(xyz)

            landmarks_points = np.array(landmarks_points)
            landmarks_pcd = self.__create_pcd(array=landmarks_points,
                                              colors=np.tile([1.0, 0, 0], (landmarks_points.shape[0], 1)))
            o3d.visualization.draw_geometries([landmarks_pcd, self.plane if plane_cloud else self.cloud])

        except Exception as e:
            print(f'Failed to draw point cloud: {e}')

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
            
    def save(self, filename):
        o3d.io.write_point_cloud(f'{filename}.pcd', self.cloud)
    