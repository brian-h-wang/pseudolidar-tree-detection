import argparse
from pointcloud_annotation.labeller import Labeller
from pointcloud_annotation.zed_utils import MultiZedDataset, ZedDataset, ZedParams
from pathlib import Path
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    args = parser.parse_args()

    params = ZedParams()
    params.min_cluster_size = 1000
    params.ransac_dist = 0.7
    params.ransac_iters = 1000
    # params.max_z_cutoff = 2.0

    # dataset = ZedDataset(zed_data_path=args.data_path, params=params)
    data_path = Path(args.data_path)
    annotations_path = Path(data_path / "kitti_object" / "training")
    multi_dataset = MultiZedDataset(data_path, annotations_path, params=params)

    for dataset in multi_dataset.datasets:
        print("Processing dataset at %s" % dataset.data_path)
        if os.path.exists(Path(dataset.data_path) / "clusters.npz"):
            print("Skipping.")
            continue
        data_labeller = Labeller(dataset)
        data_labeller.run()
        data_labeller.write_clusters()
        data_labeller.write_stats(Path(dataset.data_path) / "clustering_stats.txt")
