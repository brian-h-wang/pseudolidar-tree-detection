import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from multiprocessing import Process

import random


class TreePointCloud(object):

    def __init__(self, points=None, filename=None, o3d_pcd=None, load_zed_obj=False):
        """

        Parameters
        ----------
        points: ndarray
            Numpy array. Can be N x 3 (for x, y, z) or N x 6 (for x, y, z, R, G, B)
        filename: str
        """
        if points is None and filename is None and o3d_pcd is None:
            raise ValueError(
                "Must provide either 'points' or 'filename' argument to TreePointCloud constructor.")
        elif points is not None:
            # Construct from numpy array
            self._pcd = o3d.geometry.PointCloud()
            if points.shape[1] == 3:
                # Columns are x, y, z
                self._pcd.points = o3d.utility.Vector3dVector(points)
            elif points.shape[1] == 6:
                # Columns are x, y, z, r, g, b (RGB must be divided by 255 for valid values)
                self._pcd.points = o3d.utility.Vector3dVector(points[:,0:3])
                self._pcd.colors = o3d.utility.Vector3dVector(points[:,3:]/255.)
            else:
                raise ValueError("Invalid numpy array input. Must have 3 or 6 columns")

        elif filename is not None:
            if not load_zed_obj:
                # Construct from a .ply file
                self._pcd = o3d.io.read_point_cloud(str(filename))
            else:
                # Construct from a ZED .obj file
                mesh = o3d.io.read_triangle_mesh(str(filename))
                pcd = o3d.geometry.PointCloud()
                pcd.points = mesh.vertices
                pcd.colors = mesh.vertex_colors
                self._pcd = pcd
        elif o3d_pcd is not None:
            self._pcd = o3d_pcd
        self.ground_plane = None
        self.clusters = []
        self.original_pcd = o3d.geometry.PointCloud(self._pcd)
        self._visibility = {} # dictionary where keys are camera positions; values are the indices of points that are visible from that position

    @property
    def n_points(self):
        return len(self._pcd.points)

    def get_o3d(self):
        return self._pcd

    def to_numpy(self):
        return np.asarray(self._pcd.points)

    def downsample(self, downsample_voxel_size=0.01, verbose=False):
        n_points_before = self.n_points
        self._pcd = self._pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        if verbose:
            print("[INFO] Downsampled from %d points to %d points." % (n_points_before, self.n_points))

    def remove_ground(self, ransac_dist=0.5, ransac_n=3, ransac_iters=1000, plot=False, verbose=False):
        if verbose:
            print("[INFO] Removing ground plane...")
        plane_model, inliers = self._pcd.segment_plane(distance_threshold=ransac_dist,
                                                       ransac_n=ransac_n,
                                                       num_iterations=ransac_iters)
        if verbose:
            print("[INFO] Done!")

        plane_cloud = self._pcd.select_by_index(inliers)
        if plot:
            plane_cloud.paint_uniform_color([0.7, 0.7, 0.7])
        tree_cloud = self._pcd.select_by_index(inliers, invert=True)
        self.ground_plane = plane_cloud
        self._pcd = tree_cloud

        if plot:
            o3d.visualization.draw_geometries([plane_cloud, tree_cloud])

    def remove_points_above_height(self, z_cutoff=0.0, verbose=False):
        if verbose:
            print("[INFO] Removing points above height %.2f..." % z_cutoff)
        # plane_model, inliers = self._pcd.segment_plane(distance_threshold=ransac_dist,
        #                                                ransac_n=ransac_n,
        #                                                num_iterations=ransac_iters)
        points = np.asarray(self._pcd.points)
        below_cutoff = [i for i in range(points.shape[0]) if points[i,2] <= z_cutoff]
        self._pcd = self._pcd.select_by_index(below_cutoff)

    def cluster_trees(self, dbscan_eps=0.1, dbscan_min=50, min_cluster_size=100, plot=False, verbose=False):
        """
        Cluster the trees using DBSCAN.
        For best results, use remove_ground() and downsample() before clustering.

        Parameters
        ----------
        dbscan_eps
        dbscan_min
        plot
        verbose

        Returns
        -------
        clusters_list, indices_list : (PointCluster list, int list list)
            These lists are the same length.
            Gives a PointCluster object for each cluster, as well as a corresponding list of
            indices for the points that appear in that cluster.

        """
        if verbose:
            print("[INFO] Clustering points...")

        # cluster non-plane points
        labels = np.array(self._pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min,
                                                   print_progress=verbose))
        points = self.to_numpy()
        unique_labels = np.unique(labels)
        clusters = []
        indices_list = []
        for l in unique_labels:
            if l==-1: # skip outliers label
                continue
            if np.sum(labels==l) >= min_cluster_size:
                cluster_points = points[labels==l, :]
                indices = [i for i in range(points.shape[0]) if labels[i] == l]
                cluster = PointCluster(cluster_points)
                clusters.append(cluster)
                indices_list.append(indices)

        if verbose:
            print("[INFO] Done!")

        n_clusters = len(clusters)

        if verbose:
            print(f"point cloud has {n_clusters} clusters")

        if plot:
            o3d.visualization.draw_geometries([c.to_o3d() for c in clusters])

        return clusters, indices_list

    # def get_visible_points(self, camera_pose, radius=1.0):
    #     _, pt_map = self._pcd.hidden_point_removal(camera_pose, radius)
    #     return pt_map

    def get_visible_indices(self, camera_position, hidden_point_radius=1000000):
        camera_position = np.array(camera_position).reshape((3,1))
        key = tuple(camera_position.flatten())
        if key not in self._visibility:
            _, pt_map = self._pcd.hidden_point_removal(camera_position, radius=hidden_point_radius)
            self._visibility[key] = pt_map
        return self._visibility[key]

    def write_point_cloud(self, filename):
        o3d.io.write_point_cloud(filename, self._pcd)


"""
# THIS FUNCTION NOT USED CURRENTLY
def get_trees_bboxes(tree_cloud, clusters, plot=True):
    # get 3D bounding boxes around 3D points
    bboxes = []
    for points in clusters:
        v_points = o3d.utility.Vector3dVector(points)
        bboxes.append(o3d.geometry.OrientedBoundingBox.create_from_points(v_points))

    if plot:
        o3d.visualization.draw_geometries([tree_cloud] + bboxes)

    #TODO: remove boxes resulting from noise by checking orientation / size

    return bboxes
"""

class PointCluster(object):

    def __init__(self, points, color=None):
        """

        Parameters
        ----------
        points: ndarray
            3D points as an (N x 3) array.
        color: ndarray
            Color, specified as [R,G,B], values from 0 to 1.0.
            If not provided, a random color will be assigned.
        """
        points = np.array(points)
        points = points.reshape((-1, 3))
        self.points = points

        if color is None:
            color = [max(random.random(), 0.2), max(random.random(), 0.2), max(random.random(), 0.2)]
        self.color = np.array(color)

    def to_o3d(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.paint_uniform_color(self.color)
        return pcd

    @property
    def shape(self):
        return self.points.shape

    def __len__(self):
        return self.points.shape[0]

    def get_visible_points(self, camera_pose, radius=100000):
        pcd = self.to_o3d()
        _, pt_map = pcd.hidden_point_removal(camera_pose, radius)
        return pt_map

    def get_bounding_box(self):
        v_points = o3d.utility.Vector3dVector(self.points)
        return o3d.geometry.AxisAlignedBoundingBox.create_from_points(v_points)

    def get_bounding_box_fit_to_max_height(self, pcd):
        """
        Calculate a 3D bounding box with the top set to the highest z-coordinate of all points that
        lie within the cluster (from a top-down view).

        Parameters
        ----------
        pcd: o3d.geometry.PointCloud
            The full point cloud.

        Returns
        -------
        o3d.geometry.AxisAlignedBoundingBox

        """
        return self.get_bounding_box_fit_to_pointcloud_max(pcd, axis=2)

    def get_bounding_box_fit_to_pointcloud_max(self, pcd, axis):
        """
        Creates a bounding box around this cluster, but with one side of the bounding box
        extended to the maximum point (in one direction) in the full point cloud.

        Used e.g. for taking a bounding box that fits around the trunk of a tree only,
        and extending it to the top of the tree.

        Parameters
        ----------
        pcd
        axis: int
            0, 1, or 2
            Axis along which to extend the box

        Returns
        -------

        """
        assert axis == 0 or axis == 1 or axis == 2, "axis arg must be 0, 1, or 2"
        pts = np.asarray(pcd.points)
        # Find the highest coordinate out of all points in the full point cloud
        max_all = np.max(pts[:,axis])

        # Create an "extension point" - an extra point which is the cluster centroid,
        # but set to the point cloud max value along the axis to be extended
        max_point = np.mean(self.points, axis=0)
        max_point[axis] = max_all
        pts = np.append(self.points, max_point.reshape((1,3)), axis=0)

        # Fit a bounding box around the cluster + extension point
        v_points = o3d.utility.Vector3dVector(pts)
        max_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(v_points)
        point_indices = max_box.get_point_indices_within_bounding_box(pcd.points)
        points_in_box = np.asarray(pcd.points)[point_indices, :]
        # Find the point in the full point cloud with the highest z-coordinate, that is also
        # within the bounding box (in the x- and y- axes)
        z_max_index = np.argmax(points_in_box[:,axis])
        pts = np.append(self.points, points_in_box[z_max_index,:].reshape((1,3)), axis=0)
        v_points = o3d.utility.Vector3dVector(pts)
        return o3d.geometry.AxisAlignedBoundingBox.create_from_points(v_points)

    def get_bounding_box_fit_to_pointcloud_min(self, pcd, axis):
        """
        Creates a bounding box around this cluster, but with one side of the bounding box
        extended to the maximum point (in one direction) in the full point cloud.

        Used e.g. for taking a bounding box that fits around the midsection of a tree trunk,
        and extending it down to the ground.

        Parameters
        ----------
        pcd
        axis: int
            0, 1, or 2
            Axis along which to extend the box

        Returns
        -------

        """
        assert axis == 0 or axis == 1 or axis == 2, "axis arg must be 0, 1, or 2"
        pts = np.asarray(pcd.points)
        # Find the highest coordinate out of all points in the full point cloud
        max_all = np.min(pts[:,axis])

        # Create an "extension point" - an extra point which is the cluster centroid,
        # but set to the point cloud max value along the axis to be extended
        max_point = np.mean(self.points, axis=0)
        max_point[axis] = max_all
        pts = np.append(self.points, max_point.reshape((1,3)), axis=0)

        # Fit a bounding box around the cluster + extension point
        v_points = o3d.utility.Vector3dVector(pts)
        max_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(v_points)
        point_indices = max_box.get_point_indices_within_bounding_box(pcd.points)
        points_in_box = np.asarray(pcd.points)[point_indices, :]
        # Find the point in the full point cloud with the highest z-coordinate, that is also
        # within the bounding box (in the x- and y- axes)
        z_max_index = np.argmin(points_in_box[:,axis])
        pts = np.append(self.points, points_in_box[z_max_index,:].reshape((1,3)), axis=0)
        v_points = o3d.utility.Vector3dVector(pts)
        return o3d.geometry.AxisAlignedBoundingBox.create_from_points(v_points)

    def get_bounding_box_fit_to_pointcloud_extrema(self, pcd, axis):
        """
        Creates a bounding box around this cluster,
        extended to the maximum and minimum point (in one direction) in the full point cloud.

        Used e.g. for taking a bounding box that fits around the midsection of a tree trunk,
        and extending it vertically in both directions to cover the whole tree trunk

        Parameters
        ----------
        pcd
        axis: int
            0, 1, or 2
            Axis along which to extend the box

        Returns
        -------

        """
        assert axis == 0 or axis == 1 or axis == 2, "axis arg must be 0, 1, or 2"
        pts = np.asarray(pcd.points)
        # Find the highest coordinate out of all points in the full point cloud
        max_all = np.max(pts[:,axis])
        min_all = np.min(pts[:,axis])

        # Create an "extension point" - an extra point which is the cluster centroid,
        # but set to the point cloud max value along the axis to be extended
        max_point = np.mean(self.points, axis=0)
        max_point[axis] = max_all
        min_point = np.mean(self.points, axis=0)
        min_point[axis] = min_all
        pts = np.concatenate([self.points, max_point.reshape((1,3)), min_point.reshape((1,3))], axis=0)

        # Fit a bounding box around the cluster + extension point
        v_points = o3d.utility.Vector3dVector(pts)
        max_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(v_points)
        point_indices = max_box.get_point_indices_within_bounding_box(pcd.points)
        points_in_box = np.asarray(pcd.points)[point_indices, :]
        # Find the point in the full point cloud with the highest z-coordinate, that is also
        # within the bounding box (in the x- and y- axes)
        z_max_index = np.argmax(points_in_box[:,axis])
        z_min_index = np.argmin(points_in_box[:,axis])
        pts = np.concatenate([self.points, points_in_box[z_max_index,:].reshape((1,3)),
                              points_in_box[z_min_index,:].reshape((1,3))], axis=0)
        v_points = o3d.utility.Vector3dVector(pts)
        return o3d.geometry.AxisAlignedBoundingBox.create_from_points(v_points)

def visualize(tree_cloud, clusters):
    # Visualize clusters without blocking (so skimage viewer can also be open)
    geometries = [tree_cloud] + [c.to_o3d() for c in clusters]
    p = Process(target=o3d.visualization.draw_geometries, args=(geometries,),
                daemon=True)
    p.start()




