import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud


class PointCloud(object):
    def __init__(self, array):
        self.scale_factor = self.avg(array.shape) / array.max()
        self.image = array
        self.array = self.__preprocess(array, self.scale_factor)
        self.cloud = self.__create_pcd(self.array)

    @staticmethod
    def avg(d):
        return sum(d) / len(d)

    @staticmethod
    def __preprocess(array, scale_factor):
        rgb = []
        y, x = array.shape

        for row in range(y):
            for col in range(x):
                mod_depth = array[row][col]*scale_factor
                xyz = np.array([row, col, mod_depth])
                rgb.append(xyz)

        rgb = np.array(rgb)
        print(f'Converted raw array of shape {array.shape} to RGB array of shape {rgb.shape}')
        return rgb 
    
    @staticmethod
    def __create_pcd(array) -> PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(array)
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

    def draw_cloud_landmarks(self, landmarks):
        count, dim = self.array.shape
        try: 
            if dim != 3:
                raise ValueError(f'Expected 3 dimensions but got {dim}')

            vis = o3d.visualization.Visualizer()
            vis.create_window(height=480, width=640)

            rgb = []

            for landmark in landmarks:
                mod_depth = self.image[landmark[1]][landmark[0]] * self.scale_factor
                xyz = np.array([landmark[1], landmark[0], mod_depth])
                rgb.append(xyz)

            rgb = np.array(rgb)
            pcd = self.__create_pcd(rgb)
            o3d.io.write_point_cloud("pose.pcd", pcd)

            keep_running = True
            while keep_running:
                keep_running = vis.poll_events()
                vis.update_renderer()
                vis.add_geometry(self.cloud)

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
            
    def save(self):
        # print(np.asarray(self.cloud.points))
        o3d.io.write_point_cloud("cloud.pcd", self.cloud)
    