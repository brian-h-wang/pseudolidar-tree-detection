import logging
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from pathlib import Path

# from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from pointpillars.dataset import ForestDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help="Path to the KITTI-format dataset")
parser.add_argument('--device',
                    help='device to run the pipeline',
                    default='cuda')

args = parser.parse_args()

device = args.device
print("Using device '%s'" % device)

framework = 'torch'
kitti_path = args.dataset_path

cfg_file = "cfg/pointpillars_zed_forest.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(device=device, **cfg.model)
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

