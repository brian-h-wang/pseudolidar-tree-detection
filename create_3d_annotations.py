"""
Brian Wang, bhw45@cornell.edu
create_3d_annotations.py

Create 3D bounding box annotations on point cloud data.
Given 3D point clouds,

"""

from pathlib import Path
from pointcloud_annotation.zed_utils import MultiZedDataset, ZedParams
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to data folder.")
    parser.add_argument("--output_path", help="Path to write annotations output.")
    parser.add_argument("--min_cluster_size", help="Minimum cluster size for clustering trees in the global point cloud.",
                        required=False, type=float)
    parser.add_argument("--ransac_dist", help="Inlier distance to the ground plane for RANSAC ground removal.",
                        required=False, type=float)
    parser.add_argument("--ransac_iters", help="Number of iterations to run RANSAC for ground plane removal",
                        required=False, type=float)
    parser.add_argument("--vis", help="Visualize clustering and labeling results, for debugging.",
                        action="store_true", type=bool)

    args = parser.parse_args()

    data_path = Path(args.data_path)
    annotations_path = Path(args.output_path)

    # Set parameters for the clustering/annotation pipeline
    params = ZedParams() # default params
    params.verbose = True
    if args.vis:
        params.plot = True
        params.plot_clusters = True
        params.plot_ground_removal = True
        params.plot_global_pcd = True
    if args.min_cluster_size:
        params.min_cluster_size = args.min_cluster_size
    if args.ransac_dist:
        params.ransac_dist = args.ransac_dist
    if args.ransac_iters:
        params.ransac_iters = args.ransac_iters

    # Load dataset and write bounding box annotations
    dataset = MultiZedDataset(data_path, annotations_path, params=params)
    dataset.write_kitti_annotations()
