from open3d.ml.datasets import KITTI

from open3d._ml3d.datasets.kitti import KITTISplit
from open3d._ml3d.datasets.utils import DataProcessing

class ForestDataset(KITTI):
    """
    Class for a tree detection dataset.
    Based on KITTI format dataset, with the following changes:
        - Overrides the fixed image size
    """
    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return ForestDatasetSplit(self, split=split)


class ForestDatasetSplit(KITTISplit):
    #
    # def __init__(self, dataset, split='train'):
    #     self.cfg = dataset.cfg
    #     path_list = dataset.get_split_list(split)
    #     # log.info("Found {} pointclouds for {}".format(len(path_list), split))
    #
    #     self.path_list = path_list
    #     self.split = split
    #     self.dataset = dataset
    #
    # def __len__(self):
    #     return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        label_path = pc_path.replace('velodyne',
                                     'label_2').replace('.bin', '.txt')
        calib_path = label_path.replace('label_2', 'calib')

        pc = self.dataset.read_lidar(pc_path)
        calib = self.dataset.read_calib(calib_path)
        label = self.dataset.read_label(label_path, calib)

        image_size = [720, 1280]

        reduced_pc = DataProcessing.remove_outside_points(
            pc, calib['world_cam'], calib['cam_img'], image_size)

        # for debugging
        # print("GET DATA")
        # print(type(reduced_pc))
        # reduced_pc = reduced_pc[0:2000,:]

        data = {
            'point': reduced_pc,
            'full_point': pc,
            'feat': None,
            'calib': calib,
            'bounding_boxes': label,
        }

        return data

# args.device = _ml3d.utils.convert_device_name(args.device)
# from open3d.ml.torch.dataloaders import TorchDataloader as DataLoader
# if framework == 'torch':
#     import open3d.ml.torch as ml3d
#     from ml3d.torch.dataloaders import TorchDataloader as Dataloader
# else:
#     import tensorflow as tf
#     import open3d.ml.tf as ml3d
#
# from ml3d.tf.dataloaders import TFDataloader as Dataloader
#
# device = 'cuda'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         if device == 'cpu':
#             tf.config.set_visible_devices([], 'GPU')
#         elif device == 'gpu':
#             tf.config.set_visible_devices(gpus[0], 'GPU')
#         else:
#             idx = device.split(':')[1]
#             tf.config.set_visible_devices(gpus[int(idx)], 'GPU')
#     except RuntimeError as e:
#         print(e)

