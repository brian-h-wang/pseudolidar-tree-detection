"""
zed_utils.py
Brian Wang [bhw45@cornell.edu]

Utilities for working with point clouds from the ZED camera.
To generate 3D detector training data, process a recorded ZED .SVO file
using the zed2-mapping code.

This code takes in as inputs:
    - The fused point cloud global map
    - The individual point clouds from each time step
    - The pose history of the camera

Use the ZED API to generate all of these.

"""
import numpy as np
from pathlib import Path
import open3d as o3d
from pointcloud.dataset import Dataset, MultiDataset, Params, CameraCalibration, Pose
import os
import yaml

class ZedParams(Params):

    def __init__(self):
        super().__init__()
        # Tree clustering parameters
        self.dbscan_eps = 0.1  #
        self.dbscan_min = 10
        self.min_cluster_size = 1000
        self.downsample_voxel_size = 0.01

        self.visibility_radius = 1000000  # used in spherical projection for open3d hidden point removal function.
                                          # A large value, e.g. 100k or 1million, seems to work well

        self._scale_factor = 1.0
        self.ransac_iters = 1000
        self.ransac_dist = 0.5


class MultiZedDataset(MultiDataset):

    def __init__(self, parent_directory, annotations_directory, params=None):
        super().__init__(parent_directory, annotations_directory, params)

    def create_dataset(self, data_directory, annotations_directory, params, counter):
        return ZedDataset(data_directory, annotations_directory, params, counter)


class ZedDataset(Dataset):
    params: ZedParams

    def __init__(self, zed_data_path, annotations_directory=None, params=None, counter=None, pcd_dir="pointcloud"):
        zed_data_path = Path(zed_data_path)
        # Set ZED-specific paths
        # Paths to RGB and depth images, from the ZED
        self.rgb_path = zed_data_path / "rgb"
        self.depth_path = zed_data_path / "depth"
        self.pcd_path = zed_data_path / pcd_dir

        # Load calibration/camera parameters
        self.calib_path = zed_data_path / "calibration.yaml"
        self.calib = self.load_calibration()

        # Contains camera pose at each time step
        self.pose_path = zed_data_path / "poses.txt"
        self.pose_history = np.loadtxt(str(self.pose_path))

        # Call parent class constructor
        super().__init__(zed_data_path, annotations_directory, params, counter,
                         load_pointcloud_from_obj=True)


    def create_default_params(self):
        return ZedParams()

    def get_path_to_global_pointcloud(self):
        return self.data_path / "map.obj"

    def get_time_steps_list(self):
        time_steps = []
        for filename in os.listdir(self.rgb_path):
            s = filename.split('.')[0] # gives just the number
            try:
                time_steps.append(int(s))
            except ValueError:
                print("[WARNING] Found invalid filename '%s'" % filename)
        return time_steps


    def load_local_pointcloud(self, i):
        img_num = self.time_steps[i]
        pcd_path = self.pcd_path / ("%d.npz" % img_num)
        local_pointcloud = load_pointcloud_from_zed_npz(pcd_path)
        return local_pointcloud

    def load_pose(self, i):
        # Get pose from the poses array. A row contains time stamp, position, and quaternion
        position = self.pose_history[i, 1:4]
        quat = self.pose_history[i, 4:]
        return Pose(position, quat)

    def load_calibration(self):
        return load_zed_calib(self.calib_path)

    def bounding_box_from_cluster(self, cluster, full_pcd):
        # Fit bounding box around lowest and highest points in the full pointcloud, within the bounding box's
        # x-y limits. This accounts for removing points below (ground plane removal) and above (leaves/sky removal)
        return cluster.get_bounding_box_fit_to_pointcloud_extrema(full_pcd, axis=2)


def load_zed_calib(calib_path):
    with open(calib_path, 'r') as yamlfile:
        # read the calibration params dictionary from yaml
        calib = yaml.load(yamlfile)
    # get width, height, fx, fy, cx, cy from yaml file
    w = calib['image_width']
    h = calib['image_height']
    fx = calib['fx']
    fy = calib['fy']
    cx = calib['cx']
    cy = calib['cy']

    return CameraCalibration(w, h, fx, fy, cx, cy)


def load_pointcloud_from_zed_npz(npz_file_path, max_range=15.0):
    npz_data = np.load(npz_file_path)
    points = npz_data["points"]
    colors = npz_data["colors"].astype(np.float) / 255

    in_range = np.sqrt(points[:,0]**2 + points[:,1]**2) < max_range
    points = points[in_range,:]
    colors = colors[in_range,:]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.remove_non_finite_points()
    return pcd

def load_pointcloud_from_zed_pointclouds(pcd_path):
    pcd = o3d.geometry.PointCloud()
    points_np = np.load(pcd_path)
    points_np = points_np.reshape((-1, 4))
    pcd.points = o3d.utility.Vector3dVector(points_np[:,0:3])
    pcd.remove_non_finite_points()
    return pcd

