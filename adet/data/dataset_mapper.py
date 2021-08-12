import copy
import logging
import os.path as osp

import numpy as np
import torch
# # USER: Debug for visualization
# import matplotlib as mpl
# import matplotlib.colors as mplc
# import matplotlib.pyplot as plt
# from detectron2.utils.colormap import random_color

# from fvcore.common.file_io import PathManager
# from PIL import Image
from pycocotools import mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode

from .augmentation import RandomCropWithInstance, AugInputList, PadTransform
from .detection_utils import (annotations_to_instances, build_augmentation,
                              transform_instance_annotations)

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )

        self.basis_loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = cfg.MODEL.BASIS_MODULE.ANN_SET
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False
        self.instance_mask_format = cfg.INPUT.MASK_FORMAT

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        # # USER: Debug for visualization
        # dpi = 200
        # image_shape = image.shape[:2]  # h, w
        # img_name = dataset_dict["file_name"].split('/')
        # num_insts = len([o for o in dataset_dict["annotations"] if o["iscrowd"] == 0])
        # colors = [random_color(rgb=True, maximum=1) for _ in range(num_insts)]

        # fig = plt.figure(figsize=[image_shape[1] / dpi, image_shape[0] / dpi], frameon=False)
        # ax = plt.Axes(fig, [0.,0.,1.,1.]);  ax.axis("off");  fig.add_axes(ax)
        # ax.imshow(image[..., ::-1]) 
        # boxes = [o["bbox"] for o in dataset_dict["annotations"] if o["iscrowd"] == 0]  # xywh
        # masks = [segmToMask(o["segmentation"], image_shape) for o in dataset_dict["annotations"] if o["iscrowd"] == 0]
        # for j, box in enumerate(boxes):
        #     x0, y0, w, h = box
        #     ax.add_patch(
        #         mpl.patches.Rectangle((x0, y0), w, h, fill=False, edgecolor=colors[j], linewidth=1, alpha=0.5, linestyle="-")
        #     )
        #     rgba = np.zeros(image_shape + (4,), dtype=np.float32)
        #     rgba[..., :3] = mplc.to_rgb(colors[j])
        #     rgba[..., 3] = (masks[j] == 1).astype(np.float32) * 0.5
        #     ax.imshow(rgba)
        # plt.savefig(f"./temp/{img_name[-3]}_{img_name[-1].split('.')[0]}_0.jpg")

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            tfm = [t for t in transforms if isinstance(t, PadTransform)]
            if len(tfm) == 1:
                tfm = tfm[0]
                dataset_dict["pad_ltrb"] = [tfm.x0, tfm.y0, tfm.x1, tfm.y1]
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # # USER: Debug for visualization
        # fig = plt.figure(figsize=[image_shape[1] / dpi, image_shape[0] / dpi], frameon=False)
        # ax = plt.Axes(fig, [0.,0.,1.,1.]);  ax.axis("off");  fig.add_axes(ax)
        # ax.imshow(image[..., ::-1]) 
        # boxes = instances.gt_boxes.tensor.numpy()
        # masks = instances.gt_masks.tensor.numpy().astype(np.uint8)
        # for j, box in enumerate(boxes):
        #     x0, y0, x1, y1 = box
        #     ax.add_patch(
        #         mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colors[j], linewidth=1, alpha=0.5, linestyle="-")
        #     )
        #     rgba = np.zeros(image_shape + (4,), dtype=np.float32)
        #     rgba[..., :3] = mplc.to_rgb(colors[j])
        #     rgba[..., 3] = (masks[j] == 1).astype(np.float32) * 0.5
        #     ax.imshow(rgba)
        # plt.savefig(f"./temp/{img_name[-3]}_{img_name[-1].split('.')[0]}_1.jpg")
        # plt.close("all")

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict


class PairDatasetMapper(DatasetMapperWithBasis):
    def __init__(self, cfg, is_train):
        super().__init__(cfg, is_train=is_train)

        self.sample_nearby_frames = cfg.MODEL.PX_VOLUME.SAMPLE_NEARBY_FRAMES

    def __call__(self, data_dicts):
        data_dicts = [copy.deepcopy(d) for d in data_dicts]
        try:
            images = [
                utils.read_image(d["file_name"], format=self.image_format)
                for d in data_dicts
            ]
        except Exception as e:
            print([d["file_name"] for d in data_dicts])
            print(e)
            raise e
        for d, img in zip(data_dicts, images):
            try:
                utils.check_image_size(d, img)
            except SizeMismatchError as e:
                expected_wh = (d["width"], d["height"])
                image_wh = (img.shape[1], img.shape[0])
                if (image_wh[1], image_wh[0]) == expected_wh:
                    print(f'transposing image {d["file_name"]}')
                    img = img.transpose(1, 0, 2)
                else:
                    raise e

        # # USER: Debug for visualization
        # dpi = 200
        # image_shape = images[0].shape[:2]  # h, w
        # fig = plt.figure(figsize=[image_shape[1] / dpi, image_shape[0] / dpi], frameon=False)
        # num_insts = [len([o for o in d["annotations"] if o["iscrowd"] == 0]) for d in data_dicts]
        # colors = [[random_color(rgb=True, maximum=1) for _ in range(n)] for n in num_insts]
        # imgs_name = [d["file_name"].split('/') for d in data_dicts]
        # for i, d in enumerate(data_dicts):
        #     ax = plt.Axes(fig, [0.,0.,1.,1.]);  ax.axis("off");  fig.add_axes(ax)
        #     ax.imshow(images[i][..., ::-1])
        #     boxes = [o["bbox"] for o in d["annotations"] if o["iscrowd"] == 0]  # xywh
        #     masks = [segmToMask(o["segmentation"], image_shape) for o in d["annotations"] if o["iscrowd"] == 0]
        #     for j, box in enumerate(boxes):
        #         x0, y0, w, h = box
        #         ax.add_patch(
        #             mpl.patches.Rectangle((x0, y0), w, h, fill=False, edgecolor=colors[i][j], linewidth=1, alpha=0.5, linestyle="-")
        #         )
        #         rgba = np.zeros(image_shape + (4,), dtype=np.float32)
        #         rgba[..., :3] = mplc.to_rgb(colors[i][j])
        #         rgba[..., 3] = (masks[j] == 1).astype(np.float32) * 0.5
        #         ax.imshow(rgba)

        #     plt.savefig(f"./temp/{imgs_name[i][-3]}_{imgs_name[i][-1].split('.')[0]}_0.jpg")
        #     fig.clear()
        # plt.close("all")

        aug_inputs = AugInputList(images)
        transforms = aug_inputs.apply_augmentations(self.augmentation)
        images = aug_inputs.images
        image_shape = images[0].shape[:2]  # h, w

        # fig = plt.figure(figsize=[image_shape[1] / dpi, image_shape[0] / dpi], frameon=False)

        for i, d in enumerate(data_dicts):
            # assert images[i].shape[:2] == image_shape, \
            #     f'image size mismatch {image_shape}, img-{d["file_name"]}: {images[i].shape[:2]}'
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            d["image"] = torch.as_tensor(
                np.ascontiguousarray(images[i].transpose(2, 0, 1))
            )

            if not self.is_train:
                d.pop("annotations", None)
                d.pop("sem_seg_file_name", None)
                d.pop("pano_seg_file_name", None)

            if "annotations" not in d:
                continue

            # USER: Modify this if you want to keep them for some reason.
            for anno in d["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in d.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )
            # instances.video_ids = torch.as_tensor(
            #     [int(d["video_id"]) for _ in annos], dtype=torch.int64
            # )
            # USER: frame id, starting from 1 in each video sequence
            instances.frame_ids = torch.as_tensor(
                [int(d["frame_id"]) for _ in annos], dtype=torch.int64
            )
            # USER: instance identity id, unique in the entire dataset
            instances.inst_ids = torch.as_tensor(
                [int(obj["inst_id"]) for obj in annos], dtype=torch.int64
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            d["instances"] = utils.filter_empty_instances(instances)

            # # USER: Debug for visualization
        #     assert len(instances) == num_insts[i] 
        #     ax = plt.Axes(fig, [0.,0.,1.,1.]);  ax.axis("off");  fig.add_axes(ax)
        #     ax.imshow(images[i][..., ::-1])
        #     boxes = instances.gt_boxes.tensor.numpy()
        #     masks = instances.gt_masks.tensor.numpy().astype(np.uint8)
        #     for j, box in enumerate(boxes):
        #         x0, y0, x1, y1 = box
        #         ax.add_patch(
        #             mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colors[i][j], linewidth=1, alpha=0.5, linestyle="-")
        #         )
        #         rgba = np.zeros(image_shape + (4,), dtype=np.float32)
        #         rgba[..., :3] = mplc.to_rgb(colors[i][j])
        #         rgba[..., 3] = (masks[j] == 1).astype(np.float32) * 0.5
        #         ax.imshow(rgba)

        #     plt.savefig(f"./temp/{imgs_name[i][-3]}_{imgs_name[i][-1].split('.')[0]}_1.jpg")
        #     fig.clear()
        # plt.close('all')

        return data_dicts
