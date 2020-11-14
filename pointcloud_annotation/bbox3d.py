import numpy as np
from pathlib import Path
import os
from pointcloud.clustering import TreePointCloud, PointCluster
import yaml
from scipy.spatial.transform import Rotation as Rotation
import open3d as o3d


class _BaseParams(object):
    dbscan_eps: float
    dbscan_min: int
    min_cluster_size: int
    plot: bool
    verbose: bool
    scale_factor: float

    def __init__(self):
        # Tree clustering parameters
        self.dbscan_eps = 0  #
        self.dbscan_min = 0
        self.min_cluster_size = 1

        self.visibility_radius = 100000  # used in spherical projection for open3d hidden point removal function.
        # A large value, e.g. 100k or 1million, seems to work well

        self.plot=False
        self.verbose=False
        self._scale_factor = 1.0

    @property
    def scale_factor(self):
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, s):
        assert type(s) == float and s > 0, "Scale factor must be a float value greater than zero."
        self._scale_factor = s

class PointCloudDataset(object):
    params: _BaseParams

    params_type = _BaseParams

    def __init__(self, data_path, annotations_directory="label_2", params=None):
        # Root path for the data, e.g. data/airsim_data/forest_1
        self.data_path = Path(data_path)

        if params is None or type(params) != self.params_type:
            print("[INFO] Using default parameters")
            self.params = _BaseParams()
        else:
            self.params = params

        # Load in the global point cloud - contains points for the whole forest world, in global frame coordinates
        pcd = self.load_global_point_cloud()

        # Create folder for annotations if it doesn't exist
        self.annotations_path = self.data_path / "kitti_object" / Path(annotations_directory)
        if not os.path.exists(self.annotations_path):
            print("[INFO] Creating annotations directory at '%s'..." % str(self.annotations_path))
            Path(self.annotations_path).mkdir(parents=True)

        self.global_pointcloud = pcd

        ## Set paths to other folders in the data

        # Contains individual point clouds seen by the camera at each time step. Points are in camera frame coordinates
        self.local_pointcloud_path = self.data_path / "coords_cam"
        # Contains camera pose at each time step
        self.pose_path = self.data_path / "pose"

        ## Get time step indices from the filenames

        self.time_steps = []
        for filename in os.listdir(self.pose_path):
            if filename.startswith("pose"):
                s = filename.split('_')[1]  # gives e.g. '001.txt'
                s = s.split('.')[0] # gives e.g. '001'
                try:
                    self.time_steps.append(int(s))
                except ValueError:
                    print("[WARNING] Found invalid filename '%s'" % filename)
        self.time_steps.sort()

    def load_global_point_cloud(self):
        raise NotImplementedError

    def load_time_step(self, i):
        """
        Load the local point cloud and camera pose at a time step
        Parameters
        ----------
        i: int
            Integer time step to load

        Returns
        -------
        local_pointcloud: numpy.ndarray
            N by 3 array of points observed by the camera at time step i
            Points are given in the camera frame
        pose: AirsimCameraPose
            Contains camera parameters, position, and orientation at time step i

        """
        # Load local point cloud
        npz = np.load(str(self.local_pointcloud_path / ("coords_cam_%03d.npz" % i)))
        # coords_cam files are saved as 4 by N arrays, where row 4 is all ones.
        local_pointcloud = npz['coords_cam'].T
        local_pointcloud = local_pointcloud[:,0:3]

        # Load camera pose
        pose = AirsimCameraPose(str(self.pose_path / ("pose_%03d.txt" % i)))

        return local_pointcloud, pose


    def cluster_global_pointcloud(self, plot=False, verbose=False):
        """
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
        pts = pcd.to_numpy()
        z_max = np.max(pts[:, 2])
        trunks_index = pts[:, 2] < self.params.trunk_z_cutoff
        pcd._pcd.points = o3d.utility.Vector3dVector(pts[trunks_index, :])

        # pcd.downsample(downsample_voxel_size=0.15, verbose=True)  # Airsim pcd has density 0.15
        # pcd.remove_ground(plot=PLOT_PCD, verbose=True, ransac_iters=4000, ransac_dist=0.5)

        # Notes on getting DBSCAN to work with airsim data:
        # Set dbscan_eps to 0.1 for RealSense data
        # For Airsim, this gives no clusters.
        # Raising to 1.0 or greater seems to reduce the number of wrong clusters in the canopies
        clusters, cluster_indices = pcd.cluster_trees(dbscan_eps=self.params.dbscan_eps,
                                                      dbscan_min=self.params.dbscan_min,
                                                      min_cluster_size=self.params.min_cluster_size,
                                                      plot=False,
                                                      verbose=self.params.verbose)
        return clusters, cluster_indices


    def write_kitti_annotations(self):
        # Generate clusters
        cluster_list, cluster_indices_list = self.cluster_global_pointcloud()

        # Process for converting a cluster in the global point cloud to a local point cloud bounding box:
        # [TO BE UPDATED IF THIS GIVES WEIRD RESULTS]
        #   - Check if at least N of the points in the cluster are visible from the camera position at time step i
        #   - If yes: convert the cluster (all points) to the camera frame, then fit an AxisAlignedBoundingBox to it

        # Check which of the global bounding boxes are visible in the local pointcloud
        # A cluster is "visible" if at least N_min_visible points in the cluster are visible from the current camera pose
        n_min_visible = 20


        for i in self.time_steps:
            local_pointcloud, pose = self.load_time_step(i)
            # Annotate this time step

            # Get indices of all points in the global point cloud which are visible from the current pose
            visible_indices = self.global_pointcloud.get_visible_indices(camera_position=pose.position)

            # Transform the full point cloud to the camera frame - needed for local points bounding boxes
            world_to_local = pose.world_to_local_transform()
            points_global = np.asarray(self.global_pointcloud.original_pcd.points) # get original point cloud (not Z-cropped)
            points_local = transform_points(points_global, world_to_local)
            full_pcd_local = o3d.geometry.PointCloud()
            full_pcd_local.points = o3d.utility.Vector3dVector(points_local)

            pcd_local = o3d.geometry.PointCloud()
            pcd_local.points = o3d.utility.Vector3dVector(local_pointcloud)

            geometries = [pcd_local]
            lines = []

            for (cluster, cluster_indices) in zip(cluster_list, cluster_indices_list):

                n_cluster_points_visible = len(np.union1d(visible_indices, cluster_indices))
                cluster_is_visible = n_cluster_points_visible >= n_min_visible

                if cluster_is_visible:
                    # Transform the points in the cluster into the camera frame
                    cluster_local = PointCluster(points=transform_points(cluster.points, world_to_local))
                    bbox3d = cluster_local.get_bounding_box_fit_to_max_height(full_pcd_local)

                    # Make sure the local point cloud contains at least one point inside the bounding box
                    # Skip otherwise
                    local_points_in_box = bbox3d.get_point_indices_within_bounding_box(pcd_local.points)
                    if len(local_points_in_box) <= n_min_visible:
                        continue

                    geometries.append(bbox3d)

                    bbox_dimensions = bbox3d.get_extent()  # x,y,z dimensions
                    bbox_center = bbox3d.get_center()

                    # Calculate bounding box height, width, length in local point cloud
                    # Point cloud has x-axis right, y forwards, z up
                    height = bbox_dimensions[2] * self.params.scale_factor
                    width = bbox_dimensions[0] * self.params.scale_factor
                    length = bbox_dimensions[1] * self.params.scale_factor

                    # Note: Airsim data has x-axis pointing right, y forwards, z up
                    # Need to convert to camera coordinates - x right, y down, z forwards
                    # open3D
                    x3d = bbox_center[0] * self.params.scale_factor
                    y3d = (-bbox_center[2] * self.params.scale_factor) + height/2  # for KITTI, location point should be at bottom of bounding box
                    z3d = bbox_center[1] * self.params.scale_factor



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
                    lines.append(label)

            if self.params.plot:
                # for visualizing the axes
                # geometries += [o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0,0,0])]
                # pts = np.array([[10.5, 2.5, 5.5], [0, 0, 0]])
                # geometries += [o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts))]
                o3d.visualization.draw_geometries(geometries)

            # Write to file
            path = self.annotations_path / ("%06d.txt" % i)
            with open(path, "w") as labelfile:
                if self.params.verbose:
                    print("Writing labels to '%s'. Annotating %d bounding boxes" % (str(path), len(lines)))
                labelfile.writelines(lines)

class AirsimCameraPose(object):

    orientation: Rotation

    def __init__(self, pose_filename):
        with open(pose_filename, 'r') as pose_file:
            pose_yaml = yaml.load(pose_file, Loader=yaml.FullLoader)
        # Load fields from YAML format
        self.cx = pose_yaml['cx']
        self.cy = pose_yaml['cy']
        self.f = pose_yaml['f']
        r = pose_yaml["orientation"]
        self.orientation = Rotation.from_quat([r['xr'], r['yr'], r['zr'], r['wr']])
        p = pose_yaml["position"]
        self.position = np.array([p['x'], p['y'], p['z']])

    def rotation_matrix(self):
        return self.orientation.as_matrix()

    def world_to_local_transform(self):
        T = np.zeros((4,4))

        R = self.rotation_matrix().T  # global to robot frame rotation
        T[0:3, 0:3] = R
        T[0:3, 3]= R.dot(-self.position.reshape((3,1))).flatten()

        T[3,3] = 1

        return T

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



