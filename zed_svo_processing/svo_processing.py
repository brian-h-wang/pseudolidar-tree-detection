import pyzed.sl as sl
import numpy as np
from pathlib import Path
import os
import yaml
from struct import pack, unpack

"""

References
----------
Video Recording and Playback: https://www.stereolabs.com/docs/video/recording/
Using the Positional Tracking API: https://www.stereolabs.com/docs/positional-tracking/using-tracking/

"""

class SVOFileProcessor(object):
    """
    Class that handles:
    - Positional tracking
    - Spatial mapping
    - Saving local point clouds
    """

    def __init__(self, svo_path, output_path, verbose=False,
                 depth_quality_map=sl.DEPTH_MODE.ULTRA, depth_quality_pointcloud=sl.DEPTH_MODE.ULTRA):
        self.svo_path = str(svo_path)
        self.output_path = Path(output_path)
        self.verbose = verbose
        self.pose_history = []  # note - use list append, not numpy arrray append, for speed

        self.depth_quality_map = depth_quality_map
        self.depth_quality_pointcloud = depth_quality_pointcloud

        if not os.path.exists(self.output_path):
            Path.mkdir(self.output_path, parents=True)




    def default_init_params(self):
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(self.svo_path)
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode (default fps: 60)
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD # Use ROS-style coordinate system
        init_params.coordinate_units = sl.UNIT.METER  # Set units in meters
        init_params.depth_maximum_distance = 40.0
        return init_params

    def process_svo_map_and_pose(self, map_file="map.obj", poses_file="poses.txt", n_frames_to_skip=1,
                                 n_frames_to_trim=0):
        """
        Process the ZED SVO file and write to file:
            - A fused point cloud map
            - The camera pose history

        This function processes the SVO file with the maximum possible depth sensing quality,
        to maximize mapping and pose estimation quality.

        Parameters
        ----------
        map_file: str
            Name of the map file to output. Should be .obj file format.
        poses_file: str
            Name of the poses file ot output. Should be .txt format.
        n_frames_to_skip: int, default 0
            If set to 2 or higher, the pose history output will skip frames. Use this to subsample the SVO output.
            For example, skip_frames=2 will include every other frame in the pose file.
            Does not affect the map creation.

        Returns
        -------
        None

        """
        # Initialize the ZED Camera object
        init_params = self.default_init_params()
        init_params.depth_mode = self.depth_quality_map
        zed = sl.Camera()
        err = zed.open(init_params)

        if err == sl.ERROR_CODE.INVALID_SVO_FILE:
            print("Error processing SVO file: '%s'" % self.svo_path)
            return

        # Initialize positional tracking
        tracking_parameters = sl.PositionalTrackingParameters()
        zed.enable_positional_tracking(tracking_parameters)

        # Initialize spatial mapping
        mapping_parameters = sl.SpatialMappingParameters()
        mapping_parameters.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
        # mapping_parameters.resolution_meter = mapping_parameters.get_resolution_preset(sl.MAPPING_RESOLUTION.MEDIUM)
        mapping_parameters.resolution_meter = mapping_parameters.get_resolution_preset(sl.MAPPING_RESOLUTION.HIGH)

        # Map at short range (3.5m) to maximize quality
        # This should reduce errors like points in the sky
        # mapping_parameters.range_meter = mapping_parameters.get_range_preset(sl.MAPPING_RANGE.SHORT)
        # mapping_parameters.range_meter = mapping_parameters.get_range_preset(sl.MAPPING_RANGE.LONG)
        mapping_parameters.range_meter = mapping_parameters.get_range_preset(sl.MAPPING_RANGE.MEDIUM)
        zed.enable_spatial_mapping(mapping_parameters)

        pose_history = []

        n_frames = zed.get_svo_number_of_frames()

        assert type(n_frames_to_skip) == int and (1 <= n_frames_to_skip < n_frames), \
            "n_frames_to_skip parameter must be an int between 1 and number of frames in the SVO."

        svo_position = 0  # Keep a separate counter instead of using zed.get_svo_position() - svo position skips sometimes
        next_frame = 0  # keep track of next frame to process, for subsampling frames

        last_frame = zed.get_svo_number_of_frames() - n_frames_to_trim

        # SVO processing loop
        exit = False
        while not exit:
            # With spatial mapping enabled, zed.grab() updates the map in the background
            err = zed.grab()
            if err == sl.ERROR_CODE.SUCCESS:

                if svo_position < last_frame and svo_position >= next_frame:
                    print("\r[Processing frame %d of %d]" % (svo_position, n_frames), end='')
                    pose_history.append(get_zed_pose(zed))
                    next_frame += n_frames_to_skip
                svo_position += 1


            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print("Mapping module: Reached end of SVO file")
                exit = True

        # Write outputs
        self.write_poses(pose_history, self.output_path / poses_file)
        self.write_map(zed, self.output_path / map_file)
        zed.close()

    def process_svo_rgb_and_pointcloud(self, rgb_directory="rgb", pointcloud_directory="pointcloud",
                                       calib_file="calibration.yaml", n_frames_to_skip=1, n_frames_to_trim=0,
                                       sparsify=True, skip_images=False):
        """
        Process the ZED SVO file and write to file:
            - Stereo point clouds, generated using performance quality
            - RGB images
            - Calibration information in yaml format

        This function generates point clouds using performance quality, to simulate running on a resource-constrained
        robot online.

        Parameters
        ----------
        rgb_directory: str
            Name of the directory to save images to. Default "rgb"
        pointcloud_directory: str
            Name of the directory to save point clouds to. Default "pointcloud"
        calib_file: str
            Name of the output calibration file, .yaml file type.
            Default "calibration.yaml"
        n_frames_to_skip: int, default 0
            If set to 2 or higher, the image/pointcloud saving will skip frames. Use this to subsample the SVO output.
            For example, skip_frames=2 will save the image/pointcloud from every other frame.
        skip_images: bool
            If True, will skip writing images (write point clouds only)

        Returns
        -------
        None

        """
        # Create output directories
        rgb_output_directory = self.output_path / rgb_directory
        pcd_output_directory = self.output_path / pointcloud_directory
        for dir in [rgb_output_directory, pcd_output_directory]:
            if not os.path.exists(dir):
                Path.mkdir(dir, parents=True)

        # Initialize the ZED Camera object
        init_params = self.default_init_params()
        init_params.depth_mode = self.depth_quality_pointcloud
        zed = sl.Camera()
        err = zed.open(init_params)

        if err == sl.ERROR_CODE.INVALID_SVO_FILE:
            print("Error processing SVO file: '%s'" % self.svo_path)
            return

        n_frames = zed.get_svo_number_of_frames()


        assert type(n_frames_to_skip) == int and (1 <= n_frames_to_skip < n_frames), \
            "n_frames_to_skip parameter must be an int between 1 and number of frames in the SVO."
        svo_position = 0  # Keep a separate counter instead of using zed.get_svo_position() - svo position skips sometimes
        image_counter = 0  # Tracks how many images have been written
        next_frame = 0  # keep track of next frame to process, for subsampling frames

        last_frame = zed.get_svo_number_of_frames() - n_frames_to_trim

        # SVO processing loop
        exit = False
        while not exit:
            err = zed.grab()
            if err == sl.ERROR_CODE.SUCCESS:

                if svo_position < last_frame and svo_position >= next_frame:
                    print("\r[Processing frame %d of %d]" % (svo_position, n_frames), end='')
                    if not skip_images:
                        self.write_rgb_image(zed, rgb_directory=rgb_output_directory, image_index=image_counter)
                    if sparsify:
                        self.write_sparse_point_cloud_npz(zed, pcd_output_directory, image_index=image_counter,
                                                          lines=128, max_range=15)
                    else:
                        self.write_point_cloud_npz(zed, pcd_output_directory, image_index=image_counter)
                    # self.write_point_cloud_binary(zed, pcd_output_directory, image_index=svo_position)
                    image_counter += 1
                    next_frame += n_frames_to_skip
                svo_position += 1


            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print("Pointcloud saver: Reached end of SVO file")
                exit = True

        # Write calibration file
        self.write_calib(zed, self.output_path / calib_file)
        zed.close()

    def write_rgb_image(self, zed, rgb_directory, image_index):
        rgb_image = sl.Mat()
        zed.retrieve_image(rgb_image, sl.VIEW.LEFT)

        # Write RGB image to file
        rgb_image.write(str(Path(rgb_directory) / ("%d.jpeg" % image_index)))

    # Currently not used
    def write_depth_image(self, zed, depth_directory, image_index):
        depth_image = sl.Mat()
        zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

        # Write depth image as numpy array to preserve float32 values (distance in meters)
        np.save(Path(depth_directory) / ("%d.npy" % image_index), depth_image.get_data())

    def write_point_cloud_npz(self, zed, pcd_output_directory, image_index):
        point_cloud = sl.Mat()
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        # get_data() gives a (720 X 1280 X 4) array
        pcd_data = point_cloud.get_data().reshape((-1, 4))

        # Remove NaN/inf values
        pcd_data = pcd_data[np.isfinite(pcd_data).any(axis=1)]

        colors = zed_rgba_to_color_array(pcd_data[:, 3])

        points = pcd_data[:,0:3]

        pcd_filename = Path(pcd_output_directory) / ("%d.npz" % image_index)
        np.savez_compressed(pcd_filename, points=points, colors=colors)

    def write_sparse_point_cloud_npz(self, zed, pcd_output_directory, image_index, lines=64, max_range=0):
        point_cloud = sl.Mat()
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        # get_data() gives a (720 X 1280 X 4) array
        pcd_data = point_cloud.get_data()

        pcd_data = pcd_data.reshape((-1,4))

        # Remove NaN/inf values
        pcd_data = pcd_data[np.isfinite(pcd_data).any(axis=1)]
        colors_f32 = pcd_data[:,3]

        points = pcd_data[:,0:3]

        # limit max range
        if max_range > 0:
            below_max = get_points_within_max_range(points, max_range)
            points = points[below_max,:]
            colors_f32 = colors_f32[below_max]

        colors = zed_rgba_to_color_array(colors_f32)
        points_sparse, colors_sparse = sparsify_points(points, colors, H=lines)

        pcd_filename = Path(pcd_output_directory) / ("%d.npz" % image_index)
        np.savez_compressed(pcd_filename, points=points_sparse, colors=colors_sparse)

    def write_point_cloud_binary(self, zed, pcd_output_directory, image_index):
        point_cloud = sl.Mat()
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        # get_data() gives a (720 X 1280 X 4) array
        pcd_data = point_cloud.get_data().reshape((-1, 4))

        # Remove NaN/inf values
        pcd_data = pcd_data[np.isfinite(pcd_data).any(axis=1)]

        colors = zed_rgba_to_color_array(pcd_data[:, 3])

        points = pcd_data[:,0:3].astype(np.float32)

        points_filename = Path(pcd_output_directory) / ("%d.bin" % image_index)
        colors_filename = Path(pcd_output_directory) / ("%d_colors.bin" % image_index)
        points.tofile(points_filename)
        colors.tofile(colors_filename)

    def write_poses(self, pose_history, pose_output_path='poses.txt', replace_last_row=True):
        pose_output_path = str(pose_output_path)
        print("Writing pose history to '%s'" % pose_output_path)
        output_array = np.array(pose_history)

        if replace_last_row:
            # H264 compressed .svo files can give NaNs or wrong values in the last row of the pose history
            # Replace last row of pose history with second-to-last row (except for time stamp)
            last = output_array.shape[0] - 1
            output_array[last,1:] = output_array[last-1,1:] # do not copy the timestamp (column 0)

        with open(pose_output_path, 'w') as output_file:
            np.savetxt(output_file, output_array, fmt='%f')

    def write_map(self, zed, map_output_path="map.obj"):
        map_output_path = str(map_output_path)
        print("Writing full spatial map to '%s'" % map_output_path)
        pointcloud = sl.FusedPointCloud()
        zed.extract_whole_spatial_map(pointcloud)
        err = pointcloud.save(str(map_output_path), typeMesh=sl.MESH_FILE_FORMAT.PLY)
        if not err:
            print("Error while saving ZED point cloud!")

    def write_calib(self, zed, calib_file_path="calib.yaml"):
        calib_file_path = str(calib_file_path)
        calib = zed.get_camera_information().calibration_parameters
        fx = calib.left_cam.fx
        fy = calib.left_cam.fy
        cx = calib.left_cam.cx
        cy = calib.left_cam.cy
        w = calib.left_cam.image_size.width
        h = calib.left_cam.image_size.height
        yaml_dict = {'fx': fx, 'fy': fy,
                     'cx': cx, 'cy': cy,
                     'image_width': w, 'image_height': h}
        with open(calib_file_path, 'w') as yaml_file:
            yaml.dump(yaml_dict, yaml_file)

def get_zed_pose(zed, verbose=False):
    """

    Parameters
    ----------
    zed: sl.Camera
        The ZED camera object
    verbose: bool
        If true, will print the translation + orientation to the console

    Returns
    -------
    list
        Pose as [time, tx, ty, tz, ox, oy, oz, ow]
        time is given in seconds.
        The camera position is [tx, ty, tz]
        And orientation is the quaternion [ox, oy, oz, ow]

    """
    # Call zed.grab() each time before calling this function
    zed_pose = sl.Pose()

    # Get the pose of the camera relative to the world frame
    state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
    # Display translation and timestamp
    py_translation = sl.Translation()
    tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
    ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
    tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
    time_nsec = zed_pose.timestamp.get_nanoseconds()
    time_sec = float(time_nsec) / 1e9
    if verbose:
        print("Translation: tx: {0}, ty:  {1}, tz:  {2}, timestamp: {3}".format(tx, ty, tz, time_sec))
    # Display orientation quaternion
    py_orientation = sl.Orientation()
    ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
    oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
    oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
    ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
    if verbose:
        print("Orientation: ox: {0}, oy:  {1}, oz: {2}, ow: {3}\n".format(ox, oy, oz, ow))
    return [time_sec, tx, ty, tz, ox, oy, oz, ow]

def zed_rgba_to_color_array(rgba_values):
    """
    Convert RGBA float32 values to an N by 3 array of RGB color values

    :param rgba_values: ndarray
    :return: ndarray
    """
    rgba_values = list(rgba_values)
    # Convert float32 RGBA values to unsigned int, then to binary
    # uint_values = [unpack('I', pack('f', rgba))[0] for rgba in rgba_values]
    # Convert uint values to binary
    # binary_values = [bin(ui)[2:] for ui in uint_values]
    binary_values = [bin(unpack('I', pack('f', rgba))[0])[2:] for rgba in rgba_values]

    # Separate out 32-bit binary representation into 4 separate 8-bit values for R,G,B,A
    # alpha = [int(b[0:8], 2) for b in binary_values]
    # blue = [int(b[8:16], 2) for b in binary_values]
    # green = [int(b[16:24], 2) for b in binary_values]
    # red = [int(b[24:], 2) for b in binary_values]

    # color_array = np.array([red, green, blue], dtype=np.uint8).T

    color_array = np.zeros((len(binary_values), 3), dtype=np.uint8)
    for i, b in enumerate(binary_values):
        color_array[i,2] = int(b[8:16], 2)
        color_array[i,1] = int(b[16:24], 2)
        color_array[i,0] = int(b[24:], 2)
    return color_array

def sparsify_points(points, colors, H=64, W=512, slice=1):
    """
    Adapted from KITTI sparsification code written by Yan Wang

    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    :param max_range: depth points above this value will be removed. set to 0 to disable
    """
    # fov = np.deg2rad(70)  # ZED 2 field of view, vertically
    fov_deg = 70

    dtheta = np.radians(fov_deg) / H
    dphi = np.radians(90.0 / W)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)

    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    # theta = np.radians(2.) - np.arcsin(z / d)
    theta = np.arcsin(z / d)
    theta_ = (theta / dtheta + H/2).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 3))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 3))
    depth_map = depth_map[depth_map[:, 0] != -1.0]

    colors_sparse = - np.ones((H,W,3))
    r, g, b = colors[:,0], colors[:,1], colors[:,2]
    colors_sparse[theta_, phi_, 0] = r
    colors_sparse[theta_, phi_, 1] = g
    colors_sparse[theta_, phi_, 2] = b
    colors_sparse = colors_sparse[0::slice, :, :]
    colors_sparse = colors_sparse.reshape((-1, 3))
    colors_sparse = colors_sparse[colors_sparse[:, 0] != -1.0]

    return depth_map, colors_sparse

def get_points_within_max_range(points, max_range):
    x, y = points[:, 0], points[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    return r <= max_range

