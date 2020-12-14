from typing import Any, Union

import numpy as np
from pathlib import Path
import os
from skimage.io import imread, imsave
import argparse

from pointcloud_annotation.zed_utils import load_pointcloud_from_zed_npz, load_zed_calib
from pointcloud_annotation.dataset import Counter

"""
Converts files from ZED format to KITTI object detection format:
    - Point clouds: Npz files to "velodyne" point cloud .bin files in KITTI format.
    - Calibration
    - Images: Copy to properly numbered .png images.
"""

def convert_all(parent_path, output_path, skip_frames=0, skip_images=False):
    parent_path = Path(parent_path)
    counter = Counter()
    subdirs = [d for d in os.listdir(parent_path) if os.path.isdir(parent_path / d)]
    subdirs.sort()
    for subdir in subdirs:
        if not Path(parent_path / subdir / "pointcloud").exists():
            # Skip non-dataset folders
            continue
        print("Converting data in '%s'." % subdir)
        convert_dataset(dataset_path=parent_path / subdir,
                        output_path=output_path,
                        counter=counter, skip_frames=skip_frames, skip_images=skip_images)


def convert_dataset(dataset_path, output_path, counter, skip_frames=0, skip_images=False):
    """
    Convert all the data in a specific dataset (e.g. "forest-straight-1")

    Parameters
    ----------
    dataset_path: Path
    counter: Counter

    Returns
    -------
    None

    """

    # Prepare input and output file paths
    rgb_path = dataset_path / "rgb"
    pcd_path = dataset_path / "pointcloud"
    calib_path = dataset_path / "calibration.yaml"

    output_path = Path(output_path)
    calib_output_directory = output_path / "calib"
    velodyne_output_directory = output_path / "velodyne"
    image_output_directory = output_path / "image_2"

    # Create any output directories that don't exist yet
    for dir in [calib_output_directory, velodyne_output_directory, image_output_directory]:
        if not os.path.exists(dir):
            dir.mkdir(parents=True)

    # Get all the numbers of time steps
    frame_numbers = []
    for filename in os.listdir(rgb_path):
        s = filename.split('.')[0]  # gives just the number
        try:
            frame_numbers.append(int(s))
        except ValueError:
            print("[WARNING] Found invalid filename '%s'" % filename)
    frame_numbers.sort()


    if skip_frames > 1:
        frame_numbers = frame_numbers[0::skip_frames]


    # Load calibration once - stays the same for all frame numbers
    calib = load_zed_calib(calib_path)

    for fn in frame_numbers:
        print("Processing dataset '%s', frame %d..." % (str(dataset_path), fn))
        output_index = counter.next()

        rgb_in_filename = rgb_path / ("%d.jpeg" % fn)
        pcd_in_filename = pcd_path / ("%d.npz" % fn)

        calib_out_filename = calib_output_directory / ("%06d.txt" % output_index)
        image_out_filename = image_output_directory / ("%06d.png" % output_index)
        velodyne_out_filename = velodyne_output_directory / ("%06d.bin" % output_index)

        # Write calibration
        calib.to_kitti_calib_file(calib_out_filename)

        # Write image
        if not skip_images:
            convert_image(rgb_in_filename, image_out_filename)

        # Write stereo pointcloud to "velodyne" data
        convert_velodyne(pcd_in_filename, velodyne_out_filename)


def convert_image(image_in_filename, image_out_filename):
    # Load image
    image = imread(image_in_filename)
    imsave(image_out_filename, image)


def convert_velodyne(pcd_in_filename, velodyne_out_filename):
    pcd = load_pointcloud_from_zed_npz(pcd_in_filename)
    # Convert to x forwards, y left, z up
    # pcd.transform([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    points = np.asarray(pcd.points)

    # Convert from camera coordinates to KITTI styles
    # airsim is x right, y forwards, z up
    # rtabmap is x right, y down, z forwards
    # KITTI is x forwards, y left, z up

    pointcloud_kitti = np.zeros((points.shape[0],4))
    pointcloud_kitti[:,0:3] = points
    pointcloud_kitti[:,3] = 1  # Set intensity values to all 1s

    with open(velodyne_out_filename, 'w') as output_file:
        out_pointcloud = pointcloud_kitti.astype(np.float32)
        out_pointcloud.tofile(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", default="data/training", help="Path to data folder.")
    # parser.add_argument("--output_path", help="Path to write annotations output.")
    parser.add_argument("--testing", action="store_true", help="Set this flag to convert the testing data split. Otherwise, converts training data.")

    args = parser.parse_args()

    skip_frames = 0
    if not args.testing:
        parent_path = "data/training"
        output_path = "data/kitti_object/training"
    else:
        parent_path = "data/testing"
        output_path = "data/kitti_object/testing"

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    convert_all(parent_path, output_path, skip_frames=skip_frames, skip_images=False)