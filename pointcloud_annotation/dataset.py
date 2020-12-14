"""
dataset.py
Brian Wang [bhw45@cornell.edu]

Contains general-purpose functions for working with 3D point cloud datasets.
The Dataset class in this module is an abstract parent class for other classes
that represent specific types of datasets (Airsim, Realsense, ZED camera, etc.).

Includes classes/functions for:
    * Loading 3D data (point clouds, camera poses)
    * Calculating 3D bounding boxes for the global point cloud
    * Calculating 3D bounding box annotations on a local point cloud, in KITTI format.

Note that rtabmap point clouds are given in camera frame. X is right, Y is down, Z is forwards.

"""
import numpy as np
from pathlib import Path
import os
from pointcloud_annotation.clustering import TreePointCloud, PointCluster
import yaml
from scipy.spatial.transform import Rotation as Rotation
import open3d as o3d
from pointcloud_annotation.line_mesh import LineMesh

from abc import ABC as AbstractBaseClass
from abc import abstractmethod

class Params(object):
    dbscan_eps: float
    dbscan_min: int
    min_cluster_size: int
    plot: bool
    plot_ground_removal: bool
    plot_clusters: bool
    verbose: bool
    scale_factor: float
    plot_global_pcd: bool
    downsample_voxel_size: float
    ransac_iters: int
    ransac_dist: float

    def __init__(self):
        # Tree clustering parameters
        self.dbscan_eps = 0.1  #
        self.dbscan_min = 10
        self.min_cluster_size = 200
        self.downsample_voxel_size = 0.05

        self.visibility_radius = 1000000  # used in spherical projection for open3d hidden point removal function.
        # A large value, e.g. 100k or 1million, seems to work well

        self.plot=False
        self.verbose=False
        self.plot_global_pcd=False
        self._scale_factor = 1.0
        self.ransac_iters = 4000
        self.ransac_dist = 0.5
        self.plot_ground_removal = False
        self.plot_clusters = False
        self.load_clusters_file = False
        self.max_z_cutoff = None
        self.labels_dir = "label_2"
        self.box_max_range = 15 # boxes beyond this range are ignored

    @property
    def scale_factor(self):
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, s):
        assert type(s) == float and s > 0, "Scale factor must be a float value greater than zero."
        self._scale_factor = s


class Counter(object):
    """
    When loading data using multiple RealsenseDataset objects,
    (e.g. using multiple data captures),
    use a Counter object to keep the numbering of the resulting annotated data consistent.
    """

    def __init__(self):
        self._count = -1

    def next(self):
        self._count += 1
        return self._count


class Dataset(AbstractBaseClass):
    params: Params

    def __init__(self, data_path, annotations_directory=None,
                 params=None, counter=None, load_pointcloud_from_obj=False):
        # Root path for the data, e.g. data/rs_forest_straight/forest-straight-1
        self.data_path = Path(data_path)

        if params is None or not issubclass(type(params), Params):
            print("[INFO] Using default dataset parameters")
            self.params = self.create_default_params()
        else:
            self.params = params

        if counter is None or type(counter) != Counter:
            counter = Counter()
        self._counter = counter

        # Create folder for annotations if it doesn't exist
        if annotations_directory is None:
            # Default is [data directory]/kitti_object/label_2
            self.annotations_path = self.data_path / Path("kitti_object") / Path("label_2")
        else:
            self.annotations_path = Path(annotations_directory)
        if not os.path.exists(self.annotations_path):
            print("[INFO] Creating annotations directory at '%s'..." % str(self.annotations_path))
            Path(self.annotations_path).mkdir(parents=True)

        # Load in the global point cloud - contains points for the world map, in global frame coordinates
        global_pcd_path = self.get_path_to_global_pointcloud()
        print("[INFO] Loading file for global point cloud from '%s'..." % global_pcd_path)
        print("[INFO] Loading ZED-style .obj file")
        self.global_pointcloud = TreePointCloud(filename=global_pcd_path, load_zed_obj=load_pointcloud_from_obj)

        ## Get time step indices from the RGB image filenames
        self.time_steps = self.get_time_steps_list()
        self.time_steps.sort()

        self.clusters_path = self.data_path / "clusters.npz"

    @abstractmethod
    def create_default_params(self):
        """
        Abstract method.
        Should take in no arguments (besides self), and return an object of type DatasetParams,
        or a subclass of DatasetParams

        Returns
        -------
        Params

        """
        pass

    @abstractmethod
    def get_path_to_global_pointcloud(self):
        """
        Abstract method.
        Should take in no arguments (besides self), and return the path to the global point cloud file
        Returns
        -------

        """
        pass

    @abstractmethod
    def get_time_steps_list(self):
        """
        Abstract method.
        Should take in no arguments (besides self), and return a list of integers containing the time step numbers
        for all images/local point clouds.

        For an Airsim dataset this will be [1, 2, 3, 4, ... , N]
        For an Rtabmap dataset, this will be different, since rtabmap does not necessarily give successive numbers
        to the time steps. So the time steps may look like e.g. [13, 15, 16, 18, 20, 23, ...]

        Returns
        -------
        list [int]

        """
        pass

    @abstractmethod
    def load_local_pointcloud(self, i):
        """
        Abstract method
        Load the local point cloud at a time step
        Parameters
        ----------
        i: int
            Index of time step to load.
            Should be an index, *not* the number in the image filename.
            e.g. if the images are [4.jpg, 8.jpg, 52.jpg],
            load_time_step(0) will load the image 4.jpg
            This distinction is important for rtabmap datasets,
            because rtabmap does not necessarily give successive numbers to the time steps in the image filenames.

        Returns
        -------
        o3d.geometry.PointCloud

        """
        pass

    @abstractmethod
    def load_pose(self, i):
        """
        Abstract method.
        Load the camera pose at a time step
        Parameters
        ----------
        i: int
            Index of time step to load.
            Should be an index, *not* the number in the image filename.
            e.g. if the images are [4.jpg, 8.jpg, 52.jpg],
            load_time_step(0) will load the image 4.jpg
            This distinction is important for rtabmap datasets,
            because rtabmap does not necessarily give successive numbers to the time steps in the image filenames.

        Returns
        -------
        pose: Pose
            Contains camera position and orientation at time step i

        """
        pass

    @abstractmethod
    def load_calibration(self):
        """
        Abstract method.
        Returns an object of class Calibration, which stores the camera calibration parameters (intrinsics)

        Returns
        -------
        CameraCalibration
        """
        pass

    @abstractmethod
    def bounding_box_from_cluster(self, cluster, full_pcd):
        """
        Abstract method.
        Converts a point cluster to a bounding box, in the local camera frame coordinates.

        Parameters
        ----------
        cluster: PointCluster
            Contains a cluster of points representing one object instance.
            Points are in the local camera frame.
        full_pcd: open3d PointCloud
            Contains the full world point cloud.
            This is used for things like setting the bottom of the bounding box to be at ground level,
             according to the full point cloud.
            Points are in the local camera frame.

        Returns
        -------
        open3D bounding box

        """
        pass

    def load_clusters(self):
        npz = np.load(self.clusters_path)
        return [PointCluster(points=points) for points in npz.values()]


    def cluster_global_pointcloud(self, plot=False, verbose=False):
        """
        TODO this can be removed later - keeping it for compatibility
        Cluster tree trunks in the global point cloud.
        This is done by removing points above a set Z-height, then clustering the remaining points using DBSCAN.

        Parameters
        ----------
        plot
        verbose

        Returns
        -------

        """
        pcd = self.global_pointcloud
        return self.cluster_pointcloud(pcd, plot, verbose)

    def cluster_pointcloud(self, pcd, plot=False, verbose=False):
        """
        Cluster tree trunks in the global point cloud.
        This is done by removing points above a set Z-height, then clustering the remaining points using DBSCAN.

        Parameters
        ----------
        pcd: TreePointCloud
        plot
        verbose

        Returns
        -------

        """
        if self.params.downsample_voxel_size > 0:
            pcd.downsample(downsample_voxel_size=self.params.downsample_voxel_size, verbose=self.params.verbose)
        elif self.params.verbose:
            print("downsample_voxel_size parameter is 0, skipping downsampling...")
        pcd.remove_ground(plot=self.params.plot_ground_removal, verbose=self.params.verbose,
                          ransac_iters=self.params.ransac_iters,
                          ransac_dist=self.params.ransac_dist)
        if self.params.max_z_cutoff is not None:
            pcd.remove_points_above_height(z_cutoff = self.params.max_z_cutoff, verbose=self.params.verbose)

        clusters, cluster_indices = pcd.cluster_trees(dbscan_eps=self.params.dbscan_eps,
                                                      dbscan_min=self.params.dbscan_min,
                                                      min_cluster_size=self.params.min_cluster_size,
                                                      plot=self.params.plot_clusters,
                                                      verbose=self.params.verbose)
        return clusters, cluster_indices


    def write_kitti_annotations(self, tracking_gt=False, clustering_baseline=False):
        """

        Parameters
        ----------
        tracking_gt: bool
            If True, will write labels in tracking format
        clustering_baseline: bool
            If True, will cluster individual local point clouds using ground plane removal + DBSCAN,
            and output cluster results as the bounding box annotations.

        Returns
        -------

        """
        # Generate clusters, or load from file
        if not clustering_baseline:
            if not self.params.load_clusters_file:
                cluster_list, _ = self.cluster_global_pointcloud()
            else:
                if self.params.verbose:
                    print("Loading clusters from %s" % self.clusters_path)
                cluster_list = self.load_clusters()
                if self.params.plot_clusters:
                    o3d.visualization.draw_geometries([c.to_o3d() for c in cluster_list])

        # Process for converting a cluster in the global point cloud to a local point cloud bounding box:
        #   - Fit a 3D bounding box around the cluster points.
        #       - This box is aligned to the camera axes, in the current time step
        #   - Check if any points in the local point cloud fall within this 3D bounding box
        #   - If yes: convert the cluster (all points) to the camera frame, then fit an AxisAlignedBoundingBox to it

        # Check which of the global bounding boxes are visible in the local pointcloud
        # A cluster is "visible" if at least N_min_visible points in the cluster are visible from the current camera pose

        points_global = np.asarray(self.global_pointcloud.original_pcd.points)  # get original point cloud

        tracking_gt_lines = []

        for i in range(0, len(self.time_steps)):

            if i < 550:
                self._counter.next()
                continue


            # Load the local point cloud and camera pose at the current time step
            pcd_local = self.load_local_pointcloud(i)
            pose = self.load_pose(i)
            # Annotate this time step

            if clustering_baseline:
                cluster_list, _ = self.cluster_pointcloud(TreePointCloud(o3d_pcd=pcd_local))
                world_to_local = np.eye(4) # no separate world and local frames
                full_pcd_local = pcd_local

            # Get indices of all points in the global point cloud which are visible from the current pose
            # visible_indices = self.global_pointcloud.get_visible_indices(camera_position=pose.position)

            else:
                # Transform the full fused point cloud to the camera frame - needed for local points bounding boxes
                world_to_local = pose.world_to_local_transform()
                points_local = transform_points(points_global, world_to_local)
                full_pcd_local = o3d.geometry.PointCloud()
                full_pcd_local.points = o3d.utility.Vector3dVector(points_local)

            geometries = [pcd_local]
            lines = []

            for (cluster_id, cluster) in enumerate(cluster_list):

                # Transform the points in the cluster into the camera frame
                cluster_local = PointCluster(points=transform_points(cluster.points, world_to_local))
                bbox3d = self.bounding_box_from_cluster(cluster_local, full_pcd_local)

                # Make sure the local point cloud contains at least one point inside the bounding box
                # Skip otherwise
                # local_points_in_box = bbox3d.get_point_indices_within_bounding_box(pcd_local.points)
                # if len(local_points_in_box) == 0:
                #     continue

                bbox_dimensions = bbox3d.get_extent()  # x,y,z dimensions
                bbox_center = bbox3d.get_center()

                # Calculate bounding box height, width, length in local point cloud
                # Point cloud has x-axis right, y forwards, z up
                # height = bbox_dimensions[1] * self.params.scale_factor
                # width = bbox_dimensions[0] * self.params.scale_factor
                # length = bbox_dimensions[2] * self.params.scale_factor

                # THIS IS USED FOR ZED ROS-STYLE COORDINATES - update other datasets to use the same format later
                height = bbox_dimensions[2] * self.params.scale_factor
                width = bbox_dimensions[1] * self.params.scale_factor
                length = bbox_dimensions[0] * self.params.scale_factor

                # KITTI 3D box annotations are in camera coordinates
                # Need to convert to camera coordinates - x right, y down, z forwards
                # open3D
                x3d = -bbox_center[1] * self.params.scale_factor
                y3d = -bbox_center[2] * self.params.scale_factor + height/2  # for KITTI, location point should be at bottom of bounding box
                z3d = bbox_center[0] * self.params.scale_factor

                # Check that bounding box is within range limit, skip if not
                # Also skip bounding boxes that are behind the camera
                if np.sqrt(x3d**2 + y3d**2) > self.params.box_max_range or z3d <= 0:
                    continue

                # Create geometry for plotting
                if self.params.plot:
                    bbox3d.color = [0.1, 1.0, 0.2] # bright green
                    geometries.append(bbox3d)
                    # geometries += LineMesh.create_from_bounding_box(bbox3d).cylinder_segments

                # Create KITTI style annotation
                object_type = 'Car'  #  placeholder, label obstacles as 'Car' for compatibility with KITTI
                truncated = 0.0 # float
                occluded = 0    # int, (0,1,2,3) [currently always 0]

                rotation_y = 0
                alpha = calc_alpha(rotation_y, x3d, z3d)

                bbox = [0, 0, 0, 0]
                dimensions = [height, width, length]
                location = [x3d, y3d, z3d]

                bbox2d_str = "%.2f %.2f %.2f %.2f" % tuple(bbox)
                bbox3d_str = "%.2f %.2f %.2f %.2f %.2f %.2f %.2f" % tuple(dimensions + location + [rotation_y])
                label = "%s %.2f %.d %.2f %s %s\n" % (object_type, truncated, occluded, alpha, bbox2d_str, bbox3d_str)
                if tracking_gt:
                    # Tracking format - add in cluster ID
                    label = "%d %s" % (cluster_id+1, label)
                lines.append(label)

            if self.params.plot:
                # for visualizing the camera position and axes
                geometries += [o3d.geometry.TriangleMesh.create_coordinate_frame()]
                if self.params.plot_global_pcd:
                    global_pts = self.global_pointcloud.to_numpy()
                    global_pts_cam_frame = transform_points(global_pts, pose.world_to_local_transform())
                    global_pcd = o3d.geometry.PointCloud()
                    global_pcd.points = o3d.utility.Vector3dVector(global_pts_cam_frame)
                    global_pcd.paint_uniform_color([0, 1, 0])
                    geometries += [global_pcd]
                o3d.visualization.draw_geometries(geometries, width=640, height=320)
                # vis = o3d.visualization.VisualizerWithEditing()
                # vis.create_window()
                # render_option: o3d.visualization.RenderOption = vis.get_render_option()
                # render_option.background_color = [0.1, 0.1, 0.1]
                # for g in geometries:
                #     vis.add_geometry(g)
                # vis.run()

            # Write to file
            path = self.annotations_path / ("%06d.txt" % self._counter.next())
            if not tracking_gt:
                with open(path, "w") as labelfile:
                    if self.params.verbose:
                        print("Writing labels to '%s'. Annotating %d bounding boxes" % (str(path), len(lines)))
                    labelfile.writelines(lines)
            else:
                for l in lines:
                    # Add frame number at the start of all object lines from this frame
                    tracking_gt_lines.append("%d %s" % (i, l))
                if self.params.verbose:
                    print("[INFO] Tracking ground truth: %d objects at frame %d" % (len(lines), i))
        if tracking_gt:
            trk_gt_path = self.annotations_path / ("tracking_gt_updated.txt")
            with open(trk_gt_path, "w") as labelfile:
                labelfile.writelines(tracking_gt_lines)


class MultiDataset(AbstractBaseClass):

    def __init__(self, parent_directory, annotations_directory, params=None):
        parent_directory = Path(parent_directory)
        annotations_directory = Path(annotations_directory) / Path(params.labels_dir)
        datasets = []
        counter = Counter()
        subdirs = [d for d in os.listdir(parent_directory) if os.path.isdir(parent_directory / d)]
        subdirs.sort()
        for subdir in subdirs:
            data_directory = parent_directory / subdir
            try:
                dataset = self.create_dataset(data_directory, annotations_directory,
                                              params, counter)
                datasets.append(dataset)
                print("Loaded dataset from '%s'." % subdir)
            except:
                print("[WARNING] Error with processing dataset at '%s', skipping..." % subdir)

        self.datasets = datasets

    @abstractmethod
    def create_dataset(self, data_directory, annotations_directory, params, counter):
        pass


    def write_kitti_annotations(self, tracking_gt=False):
        for i, dataset in enumerate(self.datasets):
            print("Processing dataset [%d / %d]" % (i+1, len(self.datasets)))
            dataset.write_kitti_annotations(tracking_gt=tracking_gt)



class Pose(object):

    orientation: Rotation

    def __init__(self, position, quaternion):
        """

        Parameters
        ----------
        position : ndarray
            Size 3
            Contains x, y, z
        quaternion : ndarray
            Size 4
            Contains quaternion in scalar-last format (same format used for Scipy rotations)
        """
        self.orientation = Rotation.from_quat(quaternion)
        self.position = np.array(position)

    def rotation_matrix(self):
        return self.orientation.as_matrix()

    def world_to_local_transform(self):
        T = np.zeros((4,4))

        R = self.rotation_matrix().T  # global to robot frame rotation
        T[0:3, 0:3] = R
        T[0:3, 3]= R.dot(-self.position.reshape((3,1))).flatten()

        T[3,3] = 1

        return T


class CameraCalibration(object):
    """
    Class for storing camera calibration parameters.

    Convert to an open3D PinholeCameraIntrinsic object using the
    Calibration.to_open3d() method.

    """

    def __init__(self, image_width, image_height, fx, fy, cx, cy):
        self.width = image_width
        self.height = image_height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def to_open3d(self):
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsics.set_intrinsics(self.width, self.height,
                                         self.fx, self.fy, self.cx, self.cy)
        return camera_intrinsics

    def to_kitti_calib_file(self, calib_output_path):
        """
        Writes the calibration info to a KITTI-style calibration file

        Parameters
        ----------
        calib_output_path: Path
            Path to the output file name.

        Returns
        -------
        None

        """
        calib_out_path = Path(calib_output_path)

        # String for projection and transformation matrices
        P = ("%f 0.0 %f 0.0 0.0 %f %f 0.0 0.0 0.0 1.0 0.0\n" % (self.fx, self.cx, self.fy, self.cy))

        # Transformation matrix just needs to convert axes from camera frame to x-forward, y-left
        # Since there's no separate camera and velodyne sensors
        R0 = np.array2string(np.eye(3).flatten(), formatter={'float_kind': lambda x: "%.1f" % x})
        R0 = R0[1:-1]  # remove brackets
        R0 = R0 + '\n'

        Tcam = '0 -1 0 0 0 0 -1 0 1 0 0 0\n'

        T = np.array2string(np.eye(4)[0:3, :].flatten(), formatter={'float_kind': lambda x: "%.1f" % x})
        T = T[1:-1]  # remove brackets
        T = T + '\n'

        lines = []
        lines.append("P0: " + P)
        lines.append("P1: " + P)
        lines.append("P2: " + P)
        lines.append("P3: " + P)
        lines.append("R0_rect: " + R0)
        lines.append("Tr_velo_to_cam: " + Tcam)
        lines.append("Tr_imu_to_velo: " + T)

        with open(calib_out_path, 'w') as outfile:
            outfile.writelines(lines)


def transform_points(points, transform):
    points = np.array(points).reshape((-1, 3))
    n_points = points.shape[0]

    points_H = np.ones((n_points, 4))
    points_H[:,0:3] = points

    points_transformed_H = (transform.dot(points_H.T)).T
    return points_transformed_H[:, 0:3]


def calc_alpha(rot_y, x, z):
    # Calculate the alpha (viewing angle) for KITTI labels
    # calculate the ry after rotation
    beta = np.arctan2(z, x)
    alpha = rot_y + beta - np.pi/2 #np.sign(beta)*np.pi/2
    return alpha




