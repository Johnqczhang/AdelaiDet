# -*- coding: utf-8 -*-
"""
Convert MOT-style dataset with txt-format annotations into COCO-style json file
Usage: python datasets/prepare_cocofied_mots.py --mots | --kitti | --ht21

Some part of codes is based on https://github.com/PeizeSun/TransTrack/blob/main/track_tools/convert_mot_to_coco.py
MOTS dataset: https://motchallenge.net/data/MOTS.zip

Author: Johnqczhang
"""
import os
import os.path as osp
import argparse
import cv2
import numpy as np
import json
import copy

from functools import reduce
from pycocotools import mask as maskUtils
# USER: Debug for visualization of annotations
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
from detectron2.utils.colormap import random_color
from adet.data.dataset_mapper import segmToMask


MOT_PATH = osp.join(osp.dirname(__file__), "mot")
ANNO_PATH = osp.join(MOT_PATH, 'annotations')
DATA_PATH = osp.join(MOT_PATH, "data")
if not osp.exists(ANNO_PATH):
    os.makedirs(ANNO_PATH)

MOT_CATEGORIES = {
    1: "pedestrian", 2: "person on vehicle",
    3: "car", 4: "bicycle", 5: "motorbike", 6: "non motorized vehicle",
    7: "static_person", 8: "distractor",
    9: "occluder", 10: "occluder on the ground", 11: "occluder full",
    12: "reflection", 13: "ignored"
}
MOTS_CATEGORIES = {1: "pedestrian", 2: "car", 10: "ignored"}
HT21_CATEGORIES = {
    1: "pedestrian", 2: "person on vehicle", 3: "static", 4: "ignored"
}


def load_ht21_txt(txt_path):
    """
    Load HT21 annotations from the txt file of a video sequence.
    Each row in the txt file contains the following fields separated by commas.
        `frame_id, identity_id, box_x1, box_y1, box_w, box_h, ignored_flag, class_id, visible_flag`

    Returns:
        objects_per_frame (dict): {frame_id: list[object_dict]}
    """
    objects_per_frame = {}
    obj_ids_per_frame = {}  # To check that no frame contains two objects with same id
    # count the number of both visible and invisible objects per category
    cat_cnt = {k: [0, 0] for k in HT21_CATEGORIES}
    print(f"Loading annotations from {txt_path}")

    annos = np.loadtxt(txt_path, dtype=np.float32, delimiter=',')
    for fields in annos:
        frame_id = int(fields[0])
        if frame_id not in objects_per_frame:
            objects_per_frame[frame_id] = []
        if frame_id not in obj_ids_per_frame:
            obj_ids_per_frame[frame_id] = set()

        obj_id = int(fields[1])
        assert obj_id != -1  # -1 for detections
        assert obj_id not in obj_ids_per_frame[frame_id], (
            f"Error: multiple objects with the same id: {obj_id} in frame-{frame_id}."
        )
        obj_ids_per_frame[frame_id].add(obj_id)

        box = [float(x) for x in fields[2:6]]  # xywh
        assert box[2] > 0 and box[3] > 0
        area = box[2] * box[3]
        # filter out invalid object with empty box (this condition is never met actually.)
        if area <= 1:
            continue

        cat_id = int(fields[7])  # class label
        assert cat_id in HT21_CATEGORIES, f"Unknown object class id: {cat_id}"

        if int(fields[6]) == 0:  # ignored object (this condition is never met actually.)
            cat_id = 4

        is_visible = int(fields[8])
        assert is_visible in [0, 1]

        cat_cnt[cat_id][is_visible] += 1

        # treat invisible objects as background
        if is_visible == 0:
            continue

        # add an `ignore` field to indicate ignored objects
        objects_per_frame[frame_id].append({
            "category_id": cat_id,
            "obj_id": obj_id,
            "bbox": box,
            "area": area,
            "iscrowd": 0,
            "ignore": 0 if cat_id == 1 else 1
        })

    # print statistics information
    cat_cnt = {HT21_CATEGORIES[k]: v for k, v in cat_cnt.items() if sum(v) > 0}
    num_frames = len(objects_per_frame)
    num_objs = len(reduce(lambda x, y : x.union(y), obj_ids_per_frame.values()))
    print(f"frames: {num_frames}, [invisible, visible] annos: {cat_cnt}, objs: {num_objs}\n")

    return objects_per_frame


def load_mots_txt(txt_path):
    """ Load MOTS annotations from the txt file of a video sequence.

    Returns:
        objects_per_frame (dict): {frame_id: list[obj1_annos, obj2_annos, ...]}
    """
    objects_per_frame = {}
    obj_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    # count the number of annotations per category in the sequence.
    cat_cnt = {k: 0 for k in MOTS_CATEGORIES}
    # count the number of object identities per category in the sequence.
    obj_cnt = {k: set() for k in MOTS_CATEGORIES}
    print(f"Loading annotations from {txt_path}")

    annos = np.loadtxt(txt_path, dtype=np.str_, delimiter=' ')
    for fields in annos:
        frame_id = int(fields[0])
        if frame_id not in objects_per_frame:
            objects_per_frame[frame_id] = []
        if frame_id not in obj_ids_per_frame:
            obj_ids_per_frame[frame_id] = set()

        obj_id = int(fields[1])
        assert obj_id not in obj_ids_per_frame[frame_id], f"Multiple objects with the same id: {obj_id} in frame-{frame_id}"
        obj_ids_per_frame[frame_id].add(obj_id)

        if obj_id == 10000:  # ignored region
            cat_id = 10
        else:
            cat_id = int(fields[2])

        assert cat_id in MOTS_CATEGORIES, f"Unknown object class id: {cat_id}"

        mask = {
            "size": [int(fields[3]), int(fields[4])],  # img_h, img_w
            "counts": str(fields[5])
        }  # binary mask in RLE format
        area = maskUtils.area(mask)  # mask area
        # filter out invalid instances with empty mask
        if area < 1:
            continue

        if frame_id not in combined_mask_per_frame:
            combined_mask_per_frame[frame_id] = mask
        else:
            overlap = maskUtils.area(maskUtils.merge([combined_mask_per_frame[frame_id], mask], intersect=True))
            assert overlap <= 0., f"Objects with overlapping masks in frame-{frame_id}"
            combined_mask_per_frame[frame_id] = maskUtils.merge([combined_mask_per_frame[frame_id], mask], intersect=False)

        """
        In the original MOTS annotation, the `category_id` of "car" and "pedestrian" is 1 and 2, respectively.
        Here, we swap the `category_id` between the two categories for the sake of consistency and compatibility.
        """
        cat_id = 1 if cat_id == 2 else 2 if cat_id == 1 else cat_id
        cat_cnt[cat_id] += 1

        # count the number of object identities per category
        if MOTS_CATEGORIES[cat_id] != "ignored":
            obj_cnt[cat_id].add(obj_id)

        # add an `ignore` field to indicate ignored objects
        objects_per_frame[frame_id].append({
            "category_id": cat_id,
            "obj_id": obj_id,
            "bbox": maskUtils.toBbox(mask).tolist(),  # enclosing bbox, fmt: xywh
            "segmentation": mask,
            "area": float(area),
            "iscrowd": 0,
            "ignore": 1 if MOTS_CATEGORIES[cat_id] == "ignored" else 0
        })

    # print statistics information
    cat_cnt = {MOTS_CATEGORIES[k]: v for k, v in cat_cnt.items() if v > 0}
    obj_cnt = {MOTS_CATEGORIES[k]: len(v) for k, v in obj_cnt.items() if len(v) > 0}
    print(f"frames: {len(objects_per_frame)}, annos: {cat_cnt}, objs: {obj_cnt}\n")

    return objects_per_frame


def load_seqs_annos_from_txt(path, load_txt_func):
    """
    Load annotations from txt files of all sequences in the given path.

    Returns:
        seqs_annos (dict): {seq_id: seq_annos}
    """
    seqs_annos = {}
    seqs_txt = sorted([
        txt for txt in os.listdir(path) if txt.endswith(".txt")
    ])
    for seq_txt in seqs_txt:
        seq_id = int(seq_txt[0:4])  # e.g., "0002.txt" -> 2
        seqs_annos[seq_id] = load_txt_func(osp.join(path, seq_txt))

    return seqs_annos


def get_cocofied_json_data(args, debug=False):
    out = {
        "info": {"description": f"{args['dataset_name']}."},
        "categories": args["categories"],
        # "videos": [],
        "images": [],
        "annotations": []
    }

    # count the number of frames in the dataset
    imgs_cnt = {"valid": 0, "total": 0}
    categories = {cat["id"]: cat["name"] for cat in args["categories"]}
    # count the number of annotated objects per category in the dataset
    annos_cnt = {c: 0 for c in categories}
    # count the number of annotated indentities per category in the dataset
    objs_dict = {c: {} for c in categories}
    # count the width, height, aspect ration, and area of boxes in each category
    boxes_cnt = {c: [] for c in categories}
    ann_id = 0  # annotation id in the entire dataset, starting from 0

    print(f"\nStart COCOfying {args['dataset_name']} with seqs: {args['seqs']}:\n")

    for seq, imgs_path in zip(args["seqs"], args["imgs_path"]):
        seq_id = int(seq[-2:])
        # dict, {frame_id: [{obj1_annos}, {obj2_annos}, ...]}
        seq_annos = args["seqs_annos"][seq_id]

        img_files = sorted([
            img for img in os.listdir(imgs_path) if img.endswith(args["imExt"])
        ])
        num_imgs = len(img_files)

        if debug:
            debug_path = osp.join(imgs_path, "../debug")
            if not osp.exists(debug_path):
                os.mkdir(debug_path)
            args["debug_path"] = debug_path

        # count the number of annotated objects per category in the current sequence.
        seq_annos_cnt = {c: 0 for c in categories}
        # count the number of annotated identities per category in the current sequence.
        seq_objs_cnt = {c: 0 for c in categories}
        # count the number of annotations per frame in the current sequence.
        img_annos_cnt = []
        img_h = img_w = 0

        for frame_id, filename in enumerate(img_files):
            # frame number annotated in the filename and the original annotations
            raw_frame_id = int(filename.split('.')[0])
            # frame with empty annotations
            if raw_frame_id not in seq_annos:
                continue
            annos_cur_frame = seq_annos[raw_frame_id]
            if len(annos_cur_frame) == 0:
                continue
            # skip frame which only contains ignored annotations
            if all(anno["ignore"] for anno in annos_cur_frame):
                continue

            # raw_frame_id starts from 0 in KITTI, and 1 in MOT
            if "kitti" in imgs_path:
                assert raw_frame_id == frame_id
                img_path = f"image_02/{seq}/{filename}"
            else:
                assert raw_frame_id == frame_id + 1
                img_path = f"{seq}/img1/{filename}"

            img = cv2.imread(osp.join(imgs_path, filename))
            if img_h > 0 and img_w > 0:
                assert img.shape[0] == img_h and img.shape[1] == img_w
            img_h, img_w = img.shape[:2]

            img_id = imgs_cnt["total"] + frame_id
            img_info = {
                "file_name": img_path,
                "frame_id": frame_id,  # image number in the video sequence, starting from 0.
                "height": img_h,
                "width": img_w,
                "id": img_id,  # image number in the entire dataset, starting from 0.
                "seq_id": seq_id,
                "num_frames": num_imgs
            }
            out["images"].append(img_info)
            img_annos_cnt.append(len(annos_cur_frame))

            if debug:
                boxes, masks = [], []

            for obj in annos_cur_frame:
                # hard copy of each object's annotation to avoid reference
                anno = copy.deepcopy(obj)
                cat_id = anno["category_id"]
                box_w, box_h = anno["bbox"][2:4]
                ignore = anno["ignore"]

                """
                When encountering a new object identity, we update the field `obj_id` with a unique number
                (starting from 0 across all categories in the entire dataset).
                However, for ignored objects, we update `obj_id = -1` to avoid sampling positive proposals
                within the region of ignored objects and to ignore them during COCO evaluation, and
                update `category_id = 1` to handle KeyError during dataset loading from the COCO json file.
                """
                if ignore:
                    if debug:
                        boxes.append(np.array(anno["bbox"] + [cat_id]))
                        if "segmentation" in obj:
                            masks.append(segmToMask(obj["segmentation"], (img_h, img_w)))
                    img_annos_cnt[-1] -= 1  # discount the number of ignored objects
                    obj_id = -1
                    anno["category_id"] = 1
                else:
                    # unique object identifier in the entire dataset: dataset_seqId_catId_objId
                    obj_uId = f"{args['dataset_name']}_{seq_id}_{cat_id}_{anno['obj_id']}"
                    if obj_uId in objs_dict[cat_id]:  # an existed object
                        obj_id = objs_dict[cat_id][obj_uId]
                    else:  # a new object
                        obj_id = sum([len(v) for v in objs_dict.values()])
                        objs_dict[cat_id][obj_uId] = obj_id
                        seq_objs_cnt[cat_id] += 1

                    annos_cnt[cat_id] += 1
                    seq_annos_cnt[cat_id] += 1
                    boxes_cnt[cat_id].append([box_w, box_h, box_w / box_h, box_w * box_h])

                anno.update(
                    image_id=img_id,
                    id=ann_id,
                    obj_id=obj_id,
                    seq_id=seq_id,
                )
                anno.pop("ignore")  # "ignore" field in COCO json file is not supported.
                out["annotations"].append(anno)
                ann_id += 1

            if not debug or len(boxes) == 0:
                continue
            dpi = 200
            fig = plt.figure(figsize=[img_w / dpi, img_h / dpi], frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.]);  ax.axis("off");  fig.add_axes(ax)
            ax.imshow(img[..., ::-1])
            boxes = np.array(boxes)
            for j, box in enumerate(boxes):
                x0, y0, w, h, cat_id = box
                color = random_color(rgb=True, maximum=1)
                ax.add_patch(
                    mpl.patches.Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=2, alpha=0.8, linestyle='-')
                )
                # ax.text(
                #     x0, y0, f"{cat_id}", color=color
                # )
                if len(masks) > 0:
                    rgba = np.zeros((img_h, img_w, 4), dtype=np.float32)
                    rgba[..., :3] = mplc.to_rgb(color)
                    rgba[..., 3] = (masks[j] == 1).astype(np.float32) * 0.5
                    ax.imshow(rgba)
            plt.savefig(osp.join(args["debug_path"], filename), dpi=dpi)
            plt.close("all")

        img_cnt = len(img_annos_cnt)
        imgs_cnt["valid"] += img_cnt
        imgs_cnt["total"] += num_imgs
        seq_annos_cnt = {categories[c]: v for c, v in seq_annos_cnt.items() if v > 0}
        seq_objs_cnt = {categories[c]: v for c, v in seq_objs_cnt.items() if v > 0}
        print(
            f"seq: {seq}, images: {img_cnt}/{num_imgs}, size: {img_w}x{img_h},",
            f"annos_per_frame: [{min(img_annos_cnt)}, {max(img_annos_cnt)}, {sum(img_annos_cnt)/img_cnt:.1f}]",
            f"annos: {seq_annos_cnt}, objs: {seq_objs_cnt}\n"
        )

    annos_cnt = {categories[c]: v for c, v in annos_cnt.items() if v > 0}
    objs_cnt = {categories[c]: len(v) for c, v in objs_dict.items() if len(v) > 0}
    print(
        f"dataset: {args['dataset_name']}\n"
        f"images: {imgs_cnt['valid']}/{imgs_cnt['total']}\n"
        f"annos: {annos_cnt}\n"
        f"objs: {objs_cnt}\n"
    )
    for c in boxes_cnt:
        if len(boxes_cnt[c]) == 0:
            continue
        boxes = np.array(boxes_cnt[c])
        print(
            f"{categories[c]} (whas): "
            f"min: {boxes.min(axis=0)}, max: {boxes.max(axis=0)}, mean: {boxes.mean(axis=0)}"
        )
    print()

    return out


def cocofy_mots_challenge():
    anno_path = osp.join(ANNO_PATH, "mots_txt")
    data_path = osp.join(DATA_PATH, "MOTS-train")

    args = {}
    args["seqs_annos"] = load_seqs_annos_from_txt(anno_path, load_mots_txt)
    args["seqs"] = sorted([
        seq for seq in os.listdir(data_path) if seq.startswith("MOTS")
    ])
    args["imgs_path"] = [
        osp.join(data_path, seq, "img1") for seq in args["seqs"]
    ]
    args["categories"] = [
        {"id": 1, "name": "pedestrian", "supercategory": "person"}
    ]
    args["dataset_name"] = "MOTSChallenge Dataset"
    args["imExt"] = ".jpg"

    json_data = get_cocofied_json_data(args)

    json_file = osp.join(ANNO_PATH, "mots_train.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"COCO-style annotations saved in {json_file}.")

    """ Split into training set and validation set."""
    seqs_id = [2, 5, 9, 11]
    for seq_id in seqs_id:
        out = {k: v for k, v in json_data.items() if k not in ["images", "annotations"]}
        out["images"] = [
            copy.deepcopy(img) for img in json_data["images"] if img["seq_id"] != seq_id
        ]
        out["annotations"] = [
            copy.deepcopy(anno) for anno in json_data["annotations"] if anno["seq_id"] != seq_id
        ]
        json_file = osp.join(ANNO_PATH, f"mots_train_{seq_id:02d}.json")
        with open(json_file, 'w') as f:
            json.dump(out, f, indent=4)
        print(f"COCO-style annotations saved in {json_file}.")

        out["images"] = [
            copy.deepcopy(img) for img in json_data["images"] if img["seq_id"] == seq_id
        ]
        out["annotations"] = [
            copy.deepcopy(anno) for anno in json_data["annotations"] if anno["seq_id"] == seq_id
        ]
        json_file = osp.join(ANNO_PATH, f"mots_val_{seq_id:02d}.json")
        with open(json_file, 'w') as f:
            json.dump(out, f, indent=4)
        print(f"COCO-style annotations saved in {json_file}.")
    print()


def cocofy_kitti_mots():
    anno_path = osp.join(ANNO_PATH, "kitti_mots_txt")
    data_path = osp.join(DATA_PATH, "kitti/training/image_02")

    args = {}
    args["seqs_annos"] = load_seqs_annos_from_txt(anno_path, load_mots_txt)
    args["seqs"] = sorted([
        seq for seq in os.listdir(data_path) if seq.startswith("00")
    ])
    args["imgs_path"] = [
        osp.join(data_path, seq) for seq in args["seqs"]
    ]
    args["categories"] = [
        {"id": 1, "name": "pedestrian", "supercategory": "person"},
        {"id": 2, "name": "car", "supercategory": "vehicle"}
    ]
    args["dataset_name"] = "KITTI-MOTS Dataset"
    args["imExt"] = ".png"

    json_data = get_cocofied_json_data(args)

    # json_file = osp.join(ANNO_PATH, "kitti_mots_train_full.json")
    # with open(json_file, 'w') as f:
    #     json.dump(json_data, f, indent=4)
    # print(f"COCO-style annotations saved in {json_file}.")

    """ Split into training set and validation set."""
    seqs_id = {
        "train": [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20],
        "val": [2, 6, 7, 8, 10, 13, 14, 16, 18]
    }
    for k, seqs in seqs_id.items():
        out = {k: v for k, v in json_data.items() if k not in ["images", "annotations"]}
        out["images"] = [
            copy.deepcopy(img) for img in json_data["images"] if img["seq_id"] in seqs
        ]
        out["annotations"] = [
            copy.deepcopy(anno) for anno in json_data["annotations"] if anno["seq_id"] in seqs
        ]
        json_file = osp.join(ANNO_PATH, f"kitti_mots_{k}.json")
        with open(json_file, 'w') as f:
            json.dump(out, f, indent=4)
        print(f"COCO-style annotations saved in {json_file}.")
    print()


def cocofy_ht21():
    anno_path = osp.join(ANNO_PATH, "ht21_txt")
    data_path = osp.join(DATA_PATH, "HT21-train")

    args = {}
    args["seqs_annos"] = load_seqs_annos_from_txt(anno_path, load_ht21_txt)
    args["seqs"] = sorted([
        seq for seq in os.listdir(data_path) if seq.startswith("HT21")
    ])
    args["imgs_path"] = [
        osp.join(data_path, seq, "img1") for seq in args["seqs"]
    ]
    args["categories"] = [
        {"id": 1, "name": "pedestrian", "supercategory": "person"}
    ]
    args["dataset_name"] = "Crowd of Heads Dataset (CroHD)"
    args["imExt"] = ".jpg"

    json_data = get_cocofied_json_data(args, debug=False)

    json_file = osp.join(ANNO_PATH, "ht21_train.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=4)
    print(f"COCO-style annotations saved in {json_file}.")

    """ Split into training set and validation set."""
    seqs_id = [1, 2, 3, 4]
    for seq_id in seqs_id:
        out = {k: v for k, v in json_data.items() if k not in ["images", "annotations"]}
        out["images"] = [
            copy.deepcopy(img) for img in json_data["images"] if img["seq_id"] != seq_id
        ]
        out["annotations"] = [
            copy.deepcopy(anno) for anno in json_data["annotations"] if anno["seq_id"] != seq_id
        ]
        json_file = osp.join(ANNO_PATH, f"ht21_train_{seq_id:02d}.json")
        with open(json_file, 'w') as f:
            json.dump(out, f, indent=4)
        print(f"COCO-style annotations saved in {json_file}.")

        out["images"] = [
            copy.deepcopy(img) for img in json_data["images"] if img["seq_id"] == seq_id
        ]
        out["annotations"] = [
            copy.deepcopy(anno) for anno in json_data["annotations"] if anno["seq_id"] == seq_id
        ]
        json_file = osp.join(ANNO_PATH, f"ht21_val_{seq_id:02d}.json")
        with open(json_file, 'w') as f:
            json.dump(out, f, indent=4)
        print(f"COCO-style annotations saved in {json_file}.")
    print()


def get_parser():
    parser = argparse.ArgumentParser(description="Convert MOT-style datasets to COCO")
    parser.add_argument(
        "--mots", action="store_true", help="cocofy MOTS-Challenge",
    )
    parser.add_argument(
        "--kitti-mots", action="store_true", help="cocofy KITTI-MOTS",
    )
    parser.add_argument(
        "--ht21", action="store_true", help="cocofy Crowd of Heads Dataset (CrpHD) for Head Tracking",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.kitti_mots:
        cocofy_kitti_mots()
    if args.mots:
        cocofy_mots_challenge()
    if args.ht21:
        cocofy_ht21()
