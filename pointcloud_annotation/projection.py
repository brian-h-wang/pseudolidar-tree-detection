import numpy as np
from scipy.spatial.transform import Rotation as R
import numba as nb
import open3d as o3d
from pointcloud_annotation.clustering import PointCluster, TreePointCloud

import time

def get_visible_points(cluster, cluster_indices, full_cloud, camera_position):
    """

    Parameters
    ----------
    cluster: PointCluster
    full_pointcloud: TreePointCloud
    camera_position

    Returns
    -------

    """
    # TODO clean up this line to get rid of _pcd access
    visible = full_cloud._pcd.select_by_index(np.intersect1d(full_cloud.get_visible_indices(camera_position), cluster_indices))
    return PointCluster(np.asarray(visible.points), color=cluster.color)

def world_to_camera_frame(cluster, camera_position, camera_quaternion, P, remove_hidden=False):
    # Transform the 3D points in the cluster to the 3D camera frame

    # Define transformation matrix from world frame to camera frame
    rot = R.from_quat(camera_quaternion)
    rot_matrix = rot.as_matrix().T
    T_world_to_camera = np.zeros((4,4))
    T_world_to_camera[0:3, 0:3] = rot_matrix
    T_world_to_camera[0:3, 3] = rot_matrix.dot(-camera_position)
    T_world_to_camera[3,3] = 1

    # Transform points
    n_points = len(cluster)
    points_W = cluster.points
    points_C = np.empty(points_W.shape)  # Camera frame 3D points
    points_C = _world_to_camera_helper(points_W, T_world_to_camera, points_C)
    return points_C

@nb.njit
def _world_to_camera_helper(points_W, T_world_to_camera, points_C):
    n_points = points_C.shape[0]
    p_W = np.array([0, 0, 0, 1.0])  # Homogeneous coordinates vector
    for i in range(n_points):
        p_W[0:3] = points_W[i,:]
        points_C[i,:] = T_world_to_camera.dot(p_W.reshape((4,1))).flatten()[0:3]
    return points_C


def project_to_image(points_C, projection_matrix):
    n_points = points_C.shape[0]
    projected = np.empty((n_points, 2))
    projected = _project_to_image_helper(points_C, projection_matrix, projected)
    return projected

@nb.njit
def _project_to_image_helper(points_C, projection_matrix, projected):
    n_points = points_C.shape[0]
    p_C = np.array([0, 0, 0, 1.0]).reshape((4,1))
    for i in range(n_points):
        p_C[0:3, :] = points_C[i,:].reshape((3,1))  # 3D point in camera frame
        p = projection_matrix.dot(p_C)
        # Divide x and y coordinates by z-coordinate to get pixel coordinates
        projected[i,0] = p[0,0] / p[2,0]
        projected[i,1] = p[1,0] / p[2,0]
    return projected

@nb.njit
def draw_projected(image, projected_points, color):
    for i in range(projected_points.shape[0]):
        # projected points are in x-y format, so index 0 is column and 1 is row
        col = projected_points[i,0]
        row = projected_points[i,1]
        # if 0+radius < row < image.shape[0]-radius and 0+radius < col < image.shape[1]-radius:
        #     image[row, col, :] = [c*255 for c in color[0:3]]
        # rs = [row-1, row, row, row, row+1]
        # cs = [col, col-1, col, col+1, col]
        rs = [row-2, row-1, row-1, row-1, row, row, row, row, row, row+1, row+1, row+1, row+2]
        cs = [col, col-1, col, col+1, col-2, col-1, col, col+1, col+2, col-1, col, col+1, col]
        for r, c in zip(rs, cs):
            if 0 < r < image.shape[0] and 0 < c < image.shape[1]:
                for j in [0,1,2]:
                    image[int(np.floor(r)), int(np.floor(c)), j] = int(color[j] * 255)

