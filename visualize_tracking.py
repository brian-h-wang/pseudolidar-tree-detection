from tracker_visualization.visualization import TrackingVisualizer
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=False, help="Path to the tracking results .txt file.")
    parser.add_argument("--pcd_dir", help="Path to a directory containing point cloud .bin files.")
    parser.add_argument("--image_path", required=False, help="Path to a directory containing images for display.")
    parser.add_argument("--gt_path", required=False, help="Path to the ground truth .txt file.")
    parser.add_argument("--det_path", required=False, help="Path to the directory containing detections.")
    parser.add_argument("--n_skip", default=1, type=int, help="How many frames to skip")
    parser.add_argument("--ax", action='store_true', help="Show coordinate frame axes in visualizer")
    parser.add_argument("--frame", type=int, required=False, help="Show single frame index.")
    args = parser.parse_args()

    vis = TrackingVisualizer(results_path=args.results_path, gt_path=args.gt_path, detections_path=args.det_path,
                             pointcloud_path=args.pcd_dir, image_path=args.image_path, fps=180,
                             n_skip=args.n_skip, show_ax=args.ax, frame=args.frame)
    try:
        vis.visualize_all()
    finally:
        # If plotting detections, show detection box ranges:
        vis.plot_ranges()