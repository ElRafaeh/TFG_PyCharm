import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import PointCloud
from utils.imager import Image
from time import perf_counter


class PointCloud3D(object):
    def __init__(self, array: np.ndarray, colored_array: np.ndarray, DEBUG=False):
        """
            Extract the PointCloud of a depth map image

            Args:
                array: DepthMap image to extract pointcloud
                colored_array: RGB image for extract colors to pointcloud, values [0...1]
        """
        self.debug = DEBUG
        self.scale_factor = self.avg(array.shape) / array.max()
        self.image = array
        self.array = self.__preprocess(array, self.scale_factor, DEBUG)
        self.colors = colored_array.reshape((-1, colored_array.shape[-1]))
        self.cloud = self.__create_pcd(self.array, self.colors)
        # self.segmented_cloud, self.plane = self.get_floor_plane(DEBUG)
        self.segmented_cloud, self.plane = self.get_segmented_cloud(DEBUG)
        

    @staticmethod
    def avg(d):
        return sum(d) / len(d)

    @staticmethod
    def __preprocess(array, scale_factor, DEBUG=False):
        height, width = array.shape
        start_time = perf_counter()
        mod_depth = array * scale_factor
        xyz = np.empty((height, width, 3))
        xyz[:, :, 0] = np.arange(height)[:, np.newaxis]
        xyz[:, :, 1] = np.arange(width)
        xyz[:, :, 2] = mod_depth
        
        if DEBUG:
            print(f'Elapsed creation cloud time: {perf_counter() - start_time:.3f}s')
            
        return xyz.reshape((-1, 3))
    
    @staticmethod
    def __create_pcd(array, colors) -> PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(array)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd
    
    @classmethod
    def get_cloud_from_image(cls, mapper, image, DEBUG=False):
        raw_depth_map = mapper.map(image, DEBUG)
        depth_map = Image.from_array(raw_depth_map)

        return cls(depth_map.image, image / 255.0, DEBUG)

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

    def get_segmented_cloud(self, DEBUG=False) -> tuple[PointCloud, np.ndarray]:
        start_time = perf_counter()
        
        cl, ind = self.cloud.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
        inlier_cloud = self.cloud.select_by_index(ind)
        
        downpcd = inlier_cloud.voxel_down_sample(voxel_size=5)
        plane_model, insiders = downpcd.segment_plane(distance_threshold=8,
                                                        ransac_n=5,
                                                        num_iterations=2000)
        insider_cloud: o3d.geometry.PointCloud = downpcd.select_by_index(insiders)        
        
        if self.debug:
            end_time = perf_counter()
            o3d.visualization.draw_geometries([insider_cloud])
            print(f'Elapsed plane segmentation time: {end_time - start_time:.3f}s')

        return insider_cloud, plane_model
    
    def get_multiple_planes(self, DEBUG=False) -> tuple[PointCloud, np.ndarray]:
        start_time = perf_counter()
        
        
        cl, ind = self.cloud.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
        inlier_cloud = self.cloud.select_by_index(ind)
        downpcd = inlier_cloud.voxel_down_sample(voxel_size=5)
                
        plane_list = []
        plane_model_list = []
        N = len(downpcd.points)
        target = downpcd
        count = 0

        while count < (1 - 0.1) * N:
            plane_model, insiders = target.segment_plane(distance_threshold=8,
                                                        ransac_n=5,
                                                        num_iterations=2000)
        
            count += len(insiders)
            insider_cloud = target.select_by_index(insiders)
            insider_cloud.paint_uniform_color(np.random.rand(1,3)[0])
            outsider_cloud = target.select_by_index(insiders, invert=True)
            plane_list.append(insider_cloud)
            plane_model_list.append(plane_model)
            target = outsider_cloud
            
        print(f'Elapsed planes segmentation time: {perf_counter() - start_time:.3f}s')

        return plane_list, np.array(plane_model_list)
    
    def get_floor_plane(self, DEBUG=False):
        plane_list, plane_model = self.get_multiple_planes(DEBUG)
        plane_index = np.argmin(plane_model[:,2])
        return plane_list[plane_index], plane_model[plane_index]

    def get_landmarks_points(self, landmarks):
        landmarks_points = []

        for landmark in landmarks:
            mod_depth = self.image[landmark[1]][landmark[0]] * self.scale_factor
            xyz = np.array([landmark[1], landmark[0], mod_depth])
            landmarks_points.append(xyz)

        return np.array(landmarks_points)

    def save(self, filename):
        o3d.io.write_point_cloud(f'{filename}.pcd', self.cloud)
    