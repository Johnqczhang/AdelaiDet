import contextlib
import io
import logging
import os
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer

from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.config import configurable
from detectron2.data import DatasetCatalog, MetadataCatalog, MapDataset, DatasetFromList
from detectron2.data.build import _train_loader_from_config, build_batch_data_loader
from detectron2.data.samplers import TrainingSampler


"""
This file contains functions to parse COCO-format mots annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["PairwiseMapDataset", "load_mots_json", "register_mots_instances"]


def register_mots_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection and segmentation.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_mots_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_mots_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO-style mots annotation format.
    Currently supports instance detection and segmentation.
    (cf. detectron2.data.datasets.coco.load_coco_json)

    Args:
        json_file (str): full path to the json file in totaltext annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    assert dataset_name is not None
    is_train = "train" in dataset_name
    meta = MetadataCatalog.get(dataset_name)
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    meta.thing_classes = thing_classes

    # In COCO, certain category ids are artificially removed,
    # and by convention they are always ignored.
    # We deal with COCO's id issue and translate
    # the category ids to contiguous ids in [0, 80).

    # It works by looking at the "categories" field in the json, therefore
    # if users' own json also have incontiguous ids, we'll
    # apply this mapping as well but print a warning.
    if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
        if "coco" not in dataset_name:
            logger.warning(
                """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
            )
    id_map = {v: i for i, v in enumerate(cat_ids)}
    meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'id': 1268,  # image id in the entire data set.
    #  'frame_id': 1,  # image id in the video sequence, starting from 1.
    #  'prev_image_id': -1,
    #  'next_image_id': 2,
    #  'video_id': 1
    #  'num_frames': 600  # number of frames of the video sequence
    # }
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    img_keys = [
        "height", "width", "frame_id", "seq_id", "num_frames"
    ]
    ann_keys = ["iscrowd", "bbox", "obj_id", "area", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        for k in img_keys:
            record[k] = img_dict[k]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'
            # Filter out ignored annotations during evaluation
            if not is_train and anno["obj_id"] == -1:
                continue

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            annotation_category_id = obj["category_id"]
            try:
                obj["category_id"] = id_map[annotation_category_id]
            except KeyError as e:
                raise KeyError(
                    f"Encountered category_id={annotation_category_id} "
                    "but this id does not exist in 'categories' of the json file."
                ) from e
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


class PairwiseMapDataset(MapDataset):
    def __init__(self, dataset, map_func):
        super().__init__(dataset, map_func)
        self.T = map_func.sample_nearby_frames

    def __getitem__(self, idx):
        cur_idx = int(idx)
        data = self._dataset[cur_idx]

        # randomly sample a frame out of nearby T frames
        start = max(-self.T, 1 - data["frame_id"])
        end = min(self.T, data["num_frames"] - data["frame_id"]) + 1
        nearby_frames_offset = [i for i in range(start, end) if i != 0]
        offset = self._rng.sample(nearby_frames_offset, k=1)[0]
        next_idx = cur_idx + offset
        next_data = self._dataset[next_idx]
        assert next_data["seq_id"] == data["seq_id"], \
            f'img-{data["image_id"]} and img-{next_data["image_id"]} are not sampled from the same video'
        assert next_data["frame_id"] == data["frame_id"] + offset, \
            f'img-{data["image_id"]} and img-{next_data["image_id"]} are not adjacent'

        return {
            "width": data["width"],
            "height": data["height"],
            "pair_data": self._map_func([data, next_data])
        }


@configurable(from_config=_train_loader_from_config)
def build_mots_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    dataset = DatasetFromList(dataset, copy=False)
    dataset = PairwiseMapDataset(dataset, mapper)
    sampler = TrainingSampler(len(dataset))
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )
