"""
Load the point cloud dataset, and calculate statistics needed for setting the detector training
configuration parameters:

    - Point cloud min and max x, y, and z coordinates.
    - Bounding box size and location distribution.
"""


import logging
from pathlib import Path
import open3d.ml as _ml3d

import open3d.ml.torch as ml3d

from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from pointpillars.dataset import ForestDataset
import numpy as np

framework = 'torch'
kitti_path = "/home/brian/Datasets/ZED2/RTJ_Dataset2/kitti_object"
# kitti_path = "/home/brian/Datasets/kitti/"

cfg_file = "cfg/pointpillars_zed_forest.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = kitti_path
dataset = ForestDataset(cfg.dataset.pop('dataset_path', None), **cfg.dataset)

train_split = dataset.get_split("train")

# To avoid using too much memory, iterate over every N-th point cloud
n_skip = 20
all_points = []
all_bounding_boxes = []

for i in range(0, len(train_split), n_skip):
    data = train_split.get_data(i)
    all_points.append(data['full_point'])
    all_bounding_boxes += [b for b in data['bounding_boxes'] if b.label_class == "Car"]

xyz_min = np.array(['inf', 'inf', 'inf'], dtype=float)
xyz_max = -1 * np.array(['inf', 'inf', 'inf'], dtype=float)

# Point cloud statistics:
#   Min and max X, Y, and Z point coordinate
for points in all_points:
    for ax in [0,1,2]:
        ax_min = np.min(points[:,ax])
        if ax_min < xyz_min[ax]:
            xyz_min[ax] = ax_min

        ax_max = np.max(points[:,ax])
        if ax_max > xyz_max[ax]:
            xyz_max[ax] = ax_max

print("XYZ min: " + str(xyz_min))
print("XYZ max: " + str(xyz_max))

# Bounding box statistics:
#   Mean and std dev of z coordinate.
#   Mean and std dev of x, y, z axis size.
n_bboxes = len(all_bounding_boxes)
bbox_centers = np.empty((n_bboxes, 3))
bbox_sizes = np.empty((n_bboxes, 3))
for i, bbox in enumerate(all_bounding_boxes):
    bbox_centers[i,:] = bbox.center
    bbox_sizes[i,:] = bbox.size

# Centers: coordinates are x, y, z in velodyne frame (vertical direction is index 2)
z_position = bbox_centers[:,2]
z_mean = np.mean(z_position).item()
z_std = np.std(z_position).item()
z_min = np.min(z_position).item()
print("Bbox Z position: Mean %.4f, std dev %.4f, min %.4f" % (z_mean, z_std, z_min))

# Sizes: dimensions are width, height, length (vertical direction is index 1)
# Need mean of all three for anchors config
for i in [0,1,2]:
    size = bbox_sizes[:,i]
    sz_mean = np.mean(size).item()
    sz_std = np.std(size).item()
    print("Bbox size in dimension %d: Mean %.4f, std dev %.4f" % (i, sz_mean, sz_std))


