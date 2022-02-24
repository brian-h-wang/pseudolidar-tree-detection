import logging
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from pathlib import Path

# from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from detector_training.dataset import ForestDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help="Path to the KITTI-format dataset")
parser.add_argument('--device',
                    help='device to run the pipeline',
                    default='cuda')
parser.add_argument("--mode", help="mode for pointRCNN: RPN or RCNN")

args = parser.parse_args()

pointrcnn_mode = str(args.mode).upper()
if pointrcnn_mode != "RPN" and pointrcnn_mode != "RCNN":
    raise ValueError("Must specify 'mode' as 'rpn' or 'rcnn' to train PointRCNN.")

device = args.device
print("Using device '%s'" % device)

framework = 'torch'
kitti_path = args.dataset_path

cfg_file = "cfg/pointrcnn_zed_forest.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

# Set RPN or RCNN mode
cfg.model['mode'] = pointrcnn_mode
print("Mode is '%s'" % pointrcnn_mode)

model = ml3d.models.PointRCNN(device=device, **cfg.model)
cfg.dataset['dataset_path'] = kitti_path
dataset = ForestDataset(cfg.dataset.pop('dataset_path', None), **cfg.dataset)

pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device=device,
                                          **cfg.pipeline)

max_epoch = cfg.pipeline.max_epoch

ckpt_path = Path("ckpt")
if not ckpt_path.exists():
    ckpt_path.mkdir(parents=True)

pipeline.run_train()

pipeline.save_ckpt(max_epoch)

