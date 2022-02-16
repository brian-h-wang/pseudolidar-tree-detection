"""
Load the KITTI dataset, run PointPillars inference on one frame,
and visualize the point cloud with bounding box results.

TODO: Shows GT and inference bounding boxes in same color.
      Opened a github issue to ask how to change bbox colors.
"""


import logging
from pathlib import Path
import open3d.ml as _ml3d

import open3d.ml.torch as ml3d

from open3d.ml.vis import Visualizer, BoundingBox3D, LabelLUT
from pointpillars.dataset import ForestDataset

framework = 'torch'
kitti_path = "/home/brian/Datasets/ZED2/RTJ_Dataset2/kitti_object"

cfg_file = "cfg/pointpillars_zed_forest.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

model = ml3d.models.PointPillars(**cfg.model)
cfg.dataset['dataset_path'] = kitti_path
# dataset = ml3d.datasets.KITTI(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
dataset = ForestDataset(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
pipeline = ml3d.pipelines.ObjectDetection(model, dataset=dataset, device="gpu", **cfg.pipeline)

# download the weights.
# import os
# ckpt_folder = "./logs/"
# os.makedirs(ckpt_folder, exist_ok=True)
# ckpt_path = ckpt_folder + "pointpillars_kitti_202012221652utc.pth"
# pointpillar_url = "https://storage.googleapis.com/open3d-releases/model-zoo/pointpillars_kitti_202012221652utc.pth"
# if not os.path.exists(ckpt_path):
#     cmd = "wget {} -O {}".format(pointpillar_url, ckpt_path)
#     os.system(cmd)

ckpt_folder = Path("logs/PointPillars_TreeDetection_torch/checkpoint")
ckpt_path = ckpt_folder / "ckpt_00005.pth"

pipeline.load_ckpt(ckpt_path=str(ckpt_path))

test_split = dataset.get_split("test")
train_split = dataset.get_split("train")
# data = test_split.get_data(0)


for i in range(len(train_split)):
    data = train_split.get_data(i)
    result = pipeline.run_inference(data)[0]
    print("PCD %d: Detected %d bounding boxes" % (i, len(result)))
data = train_split.get_data(1)

# PIPELINE FROM ORIGINAL SCRIPT
"""
ObjectDetection = _ml3d.utils.get_module("pipeline", "ObjectDetection",
                                         framework)
PointPillars = _ml3d.utils.get_module("model", "PointPillars", framework)
cfg = _ml3d.utils.Config.load_from_file(
    "ml3d/configs/pointpillars_kitti.yml")

model = PointPillars(device=args.device, **cfg.model)
dataset = KITTI(args.path_kitti)
pipeline = ObjectDetection(model, dataset, device=args.device)

# load the parameters.
pipeline.load_ckpt(ckpt_path=args.path_ckpt_pointpillars)

test_split = Dataloader(dataset=dataset.get_split('training'),
                        preprocess=model.preprocess,
                        transform=None,
                        use_cache=False,
                        shuffle=False)
data = test_split[5]['data']
"""

# run inference on a single example.
result = pipeline.run_inference(data)[0]


# minimum_conf = 0.5
# result = [bbox for bbox in result if bbox.confidence > minimum_conf]

print("Detected %d bounding boxes" % len(result))

# Show ground truth
# boxes = data['bbox_objs']
# boxes = data['bounding_boxes']
# boxes.extend(result)

# Show detections
boxes = result

vis = Visualizer()

lut = LabelLUT()
for val in sorted(dataset.label_to_names.keys()):
    lut.add_label(val, val)

# Single frame with bounding boxes
vis.visualize([{
    "name": "KITTI",
    'points': data['point']
}],
    lut,
    bounding_boxes=boxes)

# Multiple frames with bounding boxes
# TODO doesn't do anything
# vis.visualize([{
#     "name": "KITTI",
#     'points': data['point'],
#     'bounding_boxes': boxes
# }],
#     lut,
# bounding_boxes=None)


# if __name__ == '__main__':
#
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
#     )
#
#     # args = parse_args()
#     # main(args)
#