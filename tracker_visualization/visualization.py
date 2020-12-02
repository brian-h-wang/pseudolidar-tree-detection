from tracker_vis.tracking import TrackerBoundingBox, TrackerResults
import numpy as np
from pathlib import Path
import open3d as o3d
import os
import time
import matplotlib.pyplot as plt
from skimage.viewer import ImageViewer
from skimage.io import imread, imshow, imread_collection
import matplotlib.pyplot as plt

GT_BOX_COLOR = [0.4, 0.95, 0.3] # bright green
# DET_COLOR = [0.95, 0.3, 0.3] # bright red
DET_COLOR = [0.90, 0.05, 1.0] # purple

def cam_to_velo_frame(velo_points):
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    return R.dot(velo_points)

class TrackingVisualizer(object):

    def __init__(self, pointcloud_path, results_path=None, gt_path=None, detections_path=None, image_path=None,
                 fps=60,  n_skip=1, show_ax=False, frame=None):
        self.tracking_results = self.detections = self.ground_truth = None
        if results_path is not None:
            self.tracking_results = TrackerResults.load(Path(results_path))
        if detections_path is not None:
            self.detections = TrackerResults.load_from_detections(Path(detections_path), box_color=DET_COLOR)
        if gt_path is not None:
            self.ground_truth = TrackerResults.load(Path(gt_path), box_color=GT_BOX_COLOR)
        self.pointcloud_path = Path(pointcloud_path)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(height=720, width=960)
        self.pcd = o3d.geometry.PointCloud()

        self.show_images = image_path is not None
        if self.show_images:
            self.image_path = Path(image_path)
            self.images = imread_collection(str(self.image_path / "*.png"), conserve_memory=True)
            self.image_vis = o3d.visualization.Visualizer()
            self.image_vis.create_window(height=720, width=720, left=1024)

        # Load the first point cloud
        # Without doing this, the point cloud colors and visualizer zoom are weird
        points = np.fromfile(self.pointcloud_path / ("%06d.bin" % 0), dtype=np.float32)
        points = (points.reshape((-1, 4))[:,0:3])
        points = cam_to_velo_frame(points.T).T
        self.pcd.points = o3d.utility.Vector3dVector(points)

        self.vis.add_geometry(self.pcd)
        if show_ax:
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
        self.prev_bboxes = []
        self.fps = fps
        if n_skip < 1:
            n_skip = 1
        self.n_skip = n_skip

        # Adjust render options
        render_option: o3d.visualization.RenderOption = self.vis.get_render_option()
        # render_option.background_color = [0.005, 0.005, 0.005]
        # render_option.background_color = [0.95, 0.95, 0.95]
        render_option.background_color = [0.1, 0.1, 0.1]
        # render_option.point_size = 3.0
        # render_option.point_size = 0.05
        render_option.point_size = 1.5

        # Set viewpoint to camera position
        vc = self.vis.get_view_control()
        vc.set_up(np.array([0, -1, 1]))
        vc.set_front(np.array([0, -0.5, -1]))
        vc.set_lookat([0, 0, 5])
        vc.set_zoom(0.1)

        # For making scatter plot of bounding box ranges
        self.box_ranges = []
        self.box_time_steps = []
        self.single_frame = frame


    def visualize_all(self):
        if self.single_frame:
            self.visualize_frame(self.single_frame)
            self.vis.run()

        t_prev_frame = time.time()
        if self.tracking_results:
            n_frames = self.tracking_results.n_frames
        elif self.ground_truth:
            n_frames = self.ground_truth.n_frames
        elif self.detections:
            n_frames = self.detections.n_frames
        frame = 0

        while True:
            t = time.time()
            if t - t_prev_frame >= (1.0 / self.fps):
                # print("\r[Frame %d]" % (frame), end='')
                print("[Frame %d]" % (frame))
                self.visualize_frame(frame)
                frame += self.n_skip
                if frame >= n_frames:
                    break
                t_prev_frame = t
            self.update_vis()


    def visualize_frame(self, frame, block=False):
        use_line_mesh = True
        # Load points as a numpy array
        points = np.fromfile(self.pointcloud_path / ("%06d.bin" % frame), dtype=np.float32)
        if self.show_images:
            try:
                image = self.images[frame]
                img_o3d = o3d.geometry.Image(image)
                self.image_vis.clear_geometries()
                self.image_vis.add_geometry(img_o3d)
                self.image_vis.poll_events()
            except:
                pass
        points = (points.reshape((-1, 4))[:,0:3])
        points = cam_to_velo_frame(points.T).T
        self.pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.ones(points.shape) * 0.5

        bboxes = []
        if self.tracking_results is not None:
            bboxes += self.tracking_results[frame]
        if self.ground_truth is not None:
            bboxes += self.ground_truth[frame]
        if self.detections is not None:
            bboxes += self.detections[frame]

        bboxes_o3d = [bbox.to_o3d() for bbox in bboxes]

        for prev_bbox in self.prev_bboxes:
            if not use_line_mesh:
                self.vis.remove_geometry(prev_bbox, reset_bounding_box=False)
            else:
                prev_bbox.remove_line(self.vis)
        bbox: o3d.geometry.AxisAlignedBoundingBox
        line_meshes = []
        for bbox in bboxes_o3d:
            if not use_line_mesh:
                self.vis.add_geometry(bbox, reset_bounding_box=False)
                # in_box = bbox.get_point_indices_within_bounding_box(self.pcd.points)
                # colors[in_box,:] = bbox.color
            else:
                points = np.asarray(bbox.get_box_points())
                color = bbox.color
                # l = [[0,1],[0,2],[0,3],[3,6],[1,6],[3,5],[2,5],[4,5],[4,6],[1,7],[2,7],[4,7]]
                l = [[0,1],[0,2],[0,3],[1,7],[1,6],[3,6],[3,5],[2,5],[4,5],[4,7],[2,7],[4,6]]
                lines = LineMesh(points, colors=color, lines=l, radius=0.03)
                lines.add_line(self.vis)
                line_meshes.append(lines)
        if not use_line_mesh:
            self.prev_bboxes = bboxes_o3d
        else:
            self.prev_bboxes = line_meshes

        if self.detections is not None:
            self.box_ranges += [bbox.range for bbox in self.detections[frame]]
            self.box_time_steps += [frame for _ in self.detections[frame]]

        # self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.pcd)
        if block:
            self.vis.run()

    def plot_ranges(self):
        if len(self.box_ranges) == 0:
            return
        plt.plot(self.box_time_steps, self.box_ranges, '.')
        plt.xlabel("Time step")
        plt.ylabel("Bounding box range")
        plt.show()


    def update_vis(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.close()


"""Module which creates mesh lines from a line set
Open3D relies upon using glLineWidth to set line width on a LineSet
However, this method is now deprecated and not fully supporeted in newer OpenGL versions
See:
    Open3D Github Pull Request - https://github.com/intel-isl/Open3D/pull/738
    Other Framework Issues - https://github.com/openframeworks/openFrameworks/issues/3460

This module aims to solve this by converting a line into a triangular mesh (which has thickness)
The basic idea is to create a cylinder for each line segment, translate it, and then rotate it.

License: MIT

"""
import numpy as np
import open3d as o3d


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def create_from_bounding_box(bbox):
        points = np.asarray(bbox.get_box_points())
        color = bbox.color
        # l = [[0,1],[0,2],[0,3],[3,6],[1,6],[3,5],[2,5],[4,5],[4,6],[1,7],[2,7],[4,7]]
        l = [[0, 1], [0, 2], [0, 3], [1, 7], [1, 6], [3, 6], [3, 5], [2, 5], [4, 5], [4, 7], [2, 7], [4, 6]]
        return LineMesh(points, colors=color, lines=l, radius=0.03)

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder, reset_bounding_box=False)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder, reset_bounding_box=False)

