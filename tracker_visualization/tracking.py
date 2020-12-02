import numpy as np
import open3d as o3d
import os
from pathlib import Path

HEIGHT=3

class TrackerBoundingBox(object):
    """
    Represents a bounding box in the tracking results.
    """

    def __init__(self, x, y, z, h, w, l, rotation_y, track_id):
        """

        Parameters
        ----------
        x: float
        y: float
        z: float
        h: float
        w: float
        l: float
        """
        self.x = x
        self.y = y
        self.z = z
        self.height = h
        self.width = w
        self.length = l
        self.rotation_y = rotation_y
        self.track_id = track_id
        self.color = [0, 0, 0]

    @staticmethod
    def from_kitti(kitti_str):
        """
        Create a TrackerBoundingBox from KITTI format data.

        Parameters
        ----------
        kitti_str: str
            Object information, including 3D bounding box,
            in KITTI object detection format.

        Returns
        -------
        TrackerBoundingBox

        """
        split = kitti_str.split(' ')
        frame = int(split[0])
        track_id = int(split[1])
        # object_type = split[2]
        # indices 3,4,5,6,7,8 are 'truncated', 'occluded', and the four 2D bounding box coords
        h = float(split[10])
        w = float(split[11])
        l = float(split[12])
        x = float(split[13])
        y = float(split[14])
        z = float(split[15])
        # Ignore rotation_y - will be approx. zero for all bboxes
        rotation_y = float(split[5])
        return frame, TrackerBoundingBox(x=x, y=y, z=z, h=h, w=w, l=l, rotation_y=rotation_y, track_id=track_id)

    @property
    def range(self):
        return np.sqrt(self.z**2 + self.x**2)

    def distance(self, other):
        """
        Computes the distance between the center of this box and another box

        Parameters
        ----------
        other: TrackerBoundingBox

        Returns
        -------
        float
            Distance between the box centers
        """
        return np.sqrt((self.x - other.x)**2 + (self.z - other.z)**2)

    @staticmethod
    def from_kitti_detection(kitti_str):
        """
        Create a TrackerBoundingBox from KITTI object detection format data.

        Parameters
        ----------
        kitti_str: str
            Object information, including 3D bounding box,
            in KITTI object detection format.

        Returns
        -------
        TrackerBoundingBox

        """
        split = kitti_str.split(' ')
        # object_type = split[0]
        # indices 3,4,5,6,7,8 are 'truncated', 'occluded', and the four 2D bounding box coords
        h = float(split[8])
        w = float(split[9])
        l = float(split[10])
        x = float(split[11])
        y = float(split[12])
        z = float(split[13])
        # Ignore rotation_y - will be approx. zero for all bboxes
        rotation_y = float(split[14])
        return TrackerBoundingBox(x=x, y=y, z=z, h=h, w=w, l=l, rotation_y=rotation_y, track_id=0)

    def to_o3d(self, cylinder=False):
        if not cylinder:
            points = np.zeros((8,3))
            i = 0

            size = self.size # ignore length, height
            # Note: x, y, z are in camera coords
            for dx in [-size/2., size/2.]:
                for dy in [0, -HEIGHT]:
                    for dz in [-size/2., size/2]:
                        points[i,:] = [self.x + dx, self.y + dy, self.z + dz]
                        i += 1

            points_v3d = o3d.utility.Vector3dVector(points)
            bbox_o3d = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points_v3d)
            bbox_o3d.color = self.color
            bbox_o3d

        else:
            bbox_o3d: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_cylinder(radius=self.size/2, height=HEIGHT)
            R = np.zeros((3,3))
            R[0,0] = 1
            R[1,2] = 1
            R[2,1] = 1
            bbox_o3d.rotate(R, [0, 0, 0])
            bbox_o3d.translate([self.x, self.y-HEIGHT/2, self.z])
            bbox_o3d.paint_uniform_color(self.color)

        return bbox_o3d

    def __str__(self):
        return "--------------------------------------\n" \
               "Bounding box ID: %d\n" \
               "Position (x,y,z): [%.3f, %.3f, %.3f]\n" \
               "Dimensions (h,w,l): [%.3f, %.3f, %.3f]\n" \
               "--------------------------------------" % (self.track_id, self.x, self.y, self.z, self.height, self.width, self.length)

    @property
    def size(self):
        return self.width

class TrackerBoundingBoxWithVariance(TrackerBoundingBox):

    def __init__(self, x, y, z, h, w, l, rotation_y, track_id, var_x, var_z, var_size):
        """

        Parameters
        ----------
        x: float
        y: float
        z: float
        h: float
        w: float
        l: float
        """
        super().__init__(x, y, z, h, w, l, rotation_y, track_id)
        self.var_x = var_x
        self.var_z = var_z
        self.var_size = var_size


    @staticmethod
    def from_kitti(kitti_str, var_str):
        """
        Create a TrackerBoundingBox from KITTI format data.

        Parameters
        ----------
        kitti_str: str
            Object information, including 3D bounding box,
            in KITTI object detection format.

        Returns
        -------
        TrackerBoundingBox

        """
        split = kitti_str.split(' ')
        frame = int(split[0])
        track_id = int(split[1])
        # object_type = split[2]
        # indices 3,4,5,6,7,8 are 'truncated', 'occluded', and the four 2D bounding box coords
        h = float(split[10])
        w = float(split[11])
        l = float(split[12])
        x = float(split[13])
        y = float(split[14])
        z = float(split[15])
        # Ignore rotation_y - will be approx. zero for all bboxes
        rotation_y = float(split[5])

        var_split = var_str.split(' ')
        var_frame = int(var_split[0])
        var_track_id = int(var_split[1])
        assert var_frame == frame, "Error: Line of variances file does not match trk file."
        assert var_track_id == track_id, "Error: Line of variances file does not match trk file."
        var_x = float(var_split[2])
        var_z = float(var_split[3])
        var_size = float(var_split[4])
        return frame, TrackerBoundingBoxWithVariance(x=x, y=y, z=z, h=h, w=w, l=l, rotation_y=rotation_y, track_id=track_id,
                                                     var_x=var_x, var_z=var_z, var_size=var_size)


class TrackerResults(object):

    def __init__(self, box_color=None):
        self._results = {}
        self.colors = {}
        self.box_color = box_color

    def __getitem__(self, time_step):
        if time_step not in self._results.keys():
            return []
        else:
            return self._results[time_step]

    def add(self, time_step, tracker_bbox):
        assert type(time_step) == int, "Must give an int time step as input"
        assert isinstance(tracker_bbox, TrackerBoundingBox), "TrackerResults.add takes in a TrackerBoundingBox input"
        if tracker_bbox.track_id not in self.colors.keys():
            if self.box_color is None:
                # new_color = list(np.random.random(3) * 0.7 + 0.2)
                new_color = list(np.random.random(3))
                self.colors[tracker_bbox.track_id] = new_color
            else:
                self.colors[tracker_bbox.track_id] = self.box_color
        tracker_bbox.color = self.colors[tracker_bbox.track_id]
        self._results[time_step] = self[time_step] + [tracker_bbox]

    @property
    def n_frames(self):
        return np.max(list(self._results.keys()))

    @staticmethod
    def load(data_path, box_color=None):
        """
        Loads a tracking results file and creates a TrackingResults object,
        which can be used for accessing the tracker data.

        The lines of the input file should contain, in order:
        (Descriptions adapted from KITTI tracking devkit readme)

        #Values    Name      Description
        ----------------------------------------------------------------------------
           1    frame        Frame within the sequence where the object appearers
           1    track id     Unique tracking id of this object within this sequence
           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
           1    truncated    Integer (0,1,2) indicating the level of truncation.
                             Note that this is in contrast to the object detection
                             benchmark where truncation is a float in [0,1].
           1    occluded     Integer (0,1,2,3) indicating occlusion state:
                             0 = fully visible, 1 = partly occluded
                             2 = largely occluded, 3 = unknown
           1    alpha        Observation angle of object, ranging [-pi..pi]
           4    bbox         2D bounding box of object in the image (0-based index):
                             contains left, top, right, bottom pixel coordinates
           3    dimensions   3D object dimensions: height, width, length (in meters)
           3    location     3D object location x,y,z in camera coordinates (in meters)
           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
           1    score        Only for results: Float, indicating confidence in
                             detection, needed for p/r curves, higher is better.

        Parameters
        ----------
        data_path

        Returns
        -------

        """
        results = TrackerResults(box_color=box_color)
        with open(data_path, 'r') as data_file:
            lines = [l.rstrip() for l in data_file.readlines()]
        for line in lines:
            frame, bbox = TrackerBoundingBox.from_kitti(line)
            results.add(frame, bbox)
        return results

    @staticmethod
    def load_from_detections(detections_path, box_color=None):
        """
        Loads a tracking results file and creates a TrackingResults object,
        which can be used for accessing the tracker data.

        The lines of the input file should contain, in order:
        (Descriptions adapted from KITTI tracking devkit readme)

        #Values    Name      Description
        ----------------------------------------------------------------------------
           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
           1    truncated    Integer (0,1,2) indicating the level of truncation.
                             Note that this is in contrast to the object detection
                             benchmark where truncation is a float in [0,1].
           1    occluded     Integer (0,1,2,3) indicating occlusion state:
                             0 = fully visible, 1 = partly occluded
                             2 = largely occluded, 3 = unknown
           1    alpha        Observation angle of object, ranging [-pi..pi]
           4    bbox         2D bounding box of object in the image (0-based index):
                             contains left, top, right, bottom pixel coordinates
           3    dimensions   3D object dimensions: height, width, length (in meters)
           3    location     3D object location x,y,z in camera coordinates (in meters)
           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
           1    score        Only for results: Float, indicating confidence in
                             detection, needed for p/r curves, higher is better.

        Parameters
        ----------
        data_path

        Returns
        -------

        """
        detections_path = Path(detections_path)
        results = TrackerResults(box_color=box_color)
        for filename in os.listdir(detections_path):
            frame = int(filename.split('.')[0])
            with open(detections_path / filename, 'r') as detections_file:
                lines = [l.rstrip() for l in detections_file.readlines()]
            for line in lines:
                bbox = TrackerBoundingBox.from_kitti_detection(line)
                results.add(frame, bbox)

        return results

    @staticmethod
    def load_with_variance(trk_path, var_path, box_color=None):
        """
        Loads a tracking results file and a tracker estimate variances file,
        and creates a TrackingResults object,
        which can be used for accessing the tracker data.

        The lines of the input file should contain, in order:
        (Descriptions adapted from KITTI tracking devkit readme)

        #Values    Name      Description
        ----------------------------------------------------------------------------
           1    frame        Frame within the sequence where the object appearers
           1    track id     Unique tracking id of this object within this sequence
           1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
           1    truncated    Integer (0,1,2) indicating the level of truncation.
                             Note that this is in contrast to the object detection
                             benchmark where truncation is a float in [0,1].
           1    occluded     Integer (0,1,2,3) indicating occlusion state:
                             0 = fully visible, 1 = partly occluded
                             2 = largely occluded, 3 = unknown
           1    alpha        Observation angle of object, ranging [-pi..pi]
           4    bbox         2D bounding box of object in the image (0-based index):
                             contains left, top, right, bottom pixel coordinates
           3    dimensions   3D object dimensions: height, width, length (in meters)
           3    location     3D object location x,y,z in camera coordinates (in meters)
           1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
           1    score        Only for results: Float, indicating confidence in
                             detection, needed for p/r curves, higher is better.


        Variance file is:
        frame   track_id     var_x     var_z      var_size (width)


        Parameters
        ----------
        data_path

        Returns
        -------

        """
        results = TrackerResults(box_color=box_color)
        with open(trk_path, 'r') as data_file:
            lines_trk = [l.rstrip() for l in data_file.readlines()]
        with open(var_path, 'r') as data_file:
            lines_var = [l.rstrip() for l in data_file.readlines()]
        for l_trk, l_var in zip(lines_trk, lines_var):
            frame, bbox = TrackerBoundingBoxWithVariance.from_kitti(l_trk, l_var)
            results.add(frame, bbox)
        return results
