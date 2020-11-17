# Pseudolidar 3D Obstacle Detection

## Getting started

### Installation

Dependencies:
- Numpy
- Open3D 0.10 or 0.11 [[Installation instructions](http://www.open3d.org/docs/release/getting_started.html)]

For ZED SVO processing:
- ZED SDK with Python API [[Installation instructions](https://www.stereolabs.com/docs/app-development/python/install/)]

### Data preparation

```
pseudolidar-tree-detection
|- zed_svo
  |- training
    |- Dataset1.svo
    |- Dataset2.svo
    ...
  |- testing
    |- TestDataset1.svo
    |- ...
|- data
  |- training
    |- Dataset1
      |- pointcloud
      |- rgb
      |- map.obj
      |- poses.txt
      |- calibration.yaml
    |- Dataset2
      |- pointcloud
      |- ...
    ...
  |- testing
    |- TestDataset1
      |- pointcloud
      |- ...
|- pointcloud_annotation
```

## Data preparation

### Processing ZED recorded SVO files

ZED SDK and Python API are required for processing SVO files.

Add .svo files to `zed_svo/training` or `zed_svo/testing`. Then, run 

``python process_svo_files --svo_dir zed_svo/training --output_dir data/training --n_skip 5``

This processes all .svo files in the `zed_svo/training` directory, creating a for each .svo file a directory containing the following:
- `pointcloud`: Contains sparsified stereo (pseudolidar) pointclouds.
- `rgb`: Contains .jpeg RGB images.
- `map.obj`: Dense fused point cloud global map.
- `poses.txt`: Pose history file.
- `calibration.yaml`: Camera calibration parameters file.

The `n_skip` argument allows skipping every n-th frame in the SVO file, for the pointcloud and rgb outputs. 

# Work in progress features

- Integrate with other 3D object detectors in addition to PointRCNN
- Support for rosbag data from ZED and Intel RealSense