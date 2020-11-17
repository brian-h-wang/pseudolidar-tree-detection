from zed_svo_processing import SVOFileProcessor
import os
from pathlib import Path
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ZED SVO files, ")
    parser.add_argument('--svo_dir', help='Path to directory containing multiple .svo files to process')
    parser.add_argument('--output_dir', help='Path to output directory.')
    parser.add_argument('--n_skip', help="Skip every n frames in the SVO file, for point cloud + image output",
                        default=1, type=int)
    parser.add_argument('--n_trim', help="Trim this many frames from the end of the video capture, for pose, point cloud, and image outputs.",
                        default=0, type=int)
    parser.add_argument("--dense", help="Skip pseudolidar sparsification and write dense point clouds.",
                        action="store_true", type=bool, default=False)
    args = parser.parse_args()

    directory = Path(args.svo_dir)
    out_directory = Path(args.output_dir)
    n_skip = args.n_skip
    n_trim = args.n_trim
    sparsify = not args.dense

    if n_skip > 1:
        print("[INFO] SVO processing will skip every %dth frame" % n_skip)
    if n_trim > 0:
        print("[INFO] SVO processing will trim the last %dth frames" % n_trim)
    for fname in os.listdir(directory):
        # if not "12-31-25" in fname:
        #     continue
        p = Path(fname)
        if not p.suffix == '.svo':
            print("Skipping non-svo file %s" % p)
            continue

        print("Processing %s" % p)
        svo_file = directory / p

        # trim the HD720_SN... part from the filename, for the output directory
        if fname.startswith("HD720"):
            output_name = p.stem.split('_')[-1]
        else:
            output_name = p.stem

        output_directory = out_directory / output_name

        svo = SVOFileProcessor(svo_path=svo_file, output_path=output_directory)
        start_time = time.time()
        # svo.process_svo_map_and_pose(map_file="map.obj", poses_file="poses.txt", n_frames_to_skip=n_skip,
        #                              n_frames_to_trim=n_trim)
        svo.process_svo_rgb_and_pointcloud(rgb_directory="rgb", pointcloud_directory="pointcloud",
                                           calib_file="calibration.yaml", n_frames_to_skip=n_skip,
                                           sparsify=sparsify, n_frames_to_trim=n_trim)
        print("Finished, runtime was %.2f" % (time.time() - start_time))
