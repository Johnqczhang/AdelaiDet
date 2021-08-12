# -*- coding: utf-8 -*-
"""
Convert MOTS txt-format annotations into COCO-style json file
Usage: python datasets/prepare_cocofied_mots.py --mots --kitti

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

from pycocotools import mask as maskUtils


KITTI_PATH = osp.join(osp.dirname(__file__), "kitti")
MOTS_PATH = osp.join(osp.dirname(__file__), 'mots')
ANNO_PATH = osp.join(MOTS_PATH, 'annotations')
if not osp.exists(ANNO_PATH):
    os.makedirs(ANNO_PATH)

MOTS_CATEGORIES = {1: "car", 2: "pedestrian"}
COCO_CATEGORIES = {1: "pedestrian", 3: "car"}
MOTS_TO_COCO_CAT_ID_MAP = {1: 3, 2: 1}


def load_annos_from_txt(txt_path):
    """
    Load annotations from the txt file of a video sequence

    Returns:
        objects_per_frame (dict): {frame_id: list(object_dict)}
    """
    objects_per_frame = {}
    obj_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks
    print(f"Loading annotations from {txt_path}")

    annos = np.loadtxt(txt_path, dtype=np.str_, delimiter=' ')
    for fields in annos:
        frame = int(fields[0])
        if frame not in objects_per_frame:
            objects_per_frame[frame] = []
        if frame not in obj_ids_per_frame:
            obj_ids_per_frame[frame] = set()

        obj_id = int(fields[1])
        if obj_id == 10000:  # ignored region
            continue

        assert obj_id not in obj_ids_per_frame[frame], f"Multiple objects with the same id: {obj_id} in frame-{frame}"
        obj_ids_per_frame[frame].add(obj_id)

        cat_id = int(fields[2])
        # cat_id should be either 1 for "car" or 2 for "pedestrian"
        assert cat_id in MOTS_CATEGORIES.keys(), f"Unknown object class id: {cat_id}"

        mask = {
            "size": [int(fields[3]), int(fields[4])],  # img_h, img_w
            "counts": str(fields[5])
        }  # binary mask in RLE format 
        area = maskUtils.area(mask)  # mask area
        if area < 1:  # filter out invalid instances with empty mask
            continue

        if frame not in combined_mask_per_frame:
            combined_mask_per_frame[frame] = mask
        else:
            overlap = maskUtils.area(maskUtils.merge([combined_mask_per_frame[frame], mask], intersect=True))
            assert overlap <= 0., f"Objects with overlapping masks in frame-{frame}"
            combined_mask_per_frame[frame] = maskUtils.merge([combined_mask_per_frame[frame], mask], intersect=False)

        objects_per_frame[frame].append({
            # use the category_id in COCO annotation
            "category_id": MOTS_TO_COCO_CAT_ID_MAP[cat_id],
            "obj_id": obj_id,
            "inst_id": obj_id % 1000,
            "bbox": maskUtils.toBbox(mask).tolist(),  # enclosing bbox, fmt: xywh
            "segmentation": mask,
            "area": float(area),
            "iscrowd": 0
        })

    return objects_per_frame


def load_seqs_annos_from_txt(path):
    seqs_annos = {}
    seqs_txt = [
        txt for txt in os.listdir(path) if txt.endswith(".txt")
    ]
    for seq_txt in seqs_txt:
        seq_id = int(seq_txt[0:4])  # e.g., "0002.txt" -> 2
        seqs_annos[seq_id] = load_annos_from_txt(osp.join(path, seq_txt))

    return seqs_annos


def cocofy_mots(args):
    out = args["out"]
    img_cnt = 0
    imgs_cnt = 0
    ann_cnt = 0
    annos_cnt = {k: 0 for k in COCO_CATEGORIES.values()}
    inst_ids = {
        k: {"db": {}, "cnt": 0} for k in COCO_CATEGORIES.values()
    }
    obj_ids = {"db": {}, "cnt": 0}

    for seq, imgs_path in zip(args["seqs"], args["imgs_path"]):
        seq_id = int(seq[-2:])
        # dict, {frame_id: [{obj1_annos}, {obj2_annos}, ...]}
        annos_per_frame = args["seqs_annos"][seq_id]
        num_valid_frames = len([f for f in annos_per_frame if len(annos_per_frame[f]) > 0])

        imgs = sorted([
            img for img in os.listdir(imgs_path)
            if img.endswith(".jpg") or img.endswith(".png")
        ])
        num_imgs = len(imgs)
        imgs_cnt += num_imgs

        num_annos = {k: 0 for k in COCO_CATEGORIES.values()}  # count number of annotated instances
        num_insts = {k: 0 for k in COCO_CATEGORIES.values()}  # count number of identities within each category
        # image number with non-empty annotations in the video sequence, starting from 1.
        frame_id = 0

        for i, img_name in enumerate(imgs):
            real_frame_id = int(img_name.split('.')[0])
            # frame with empty annotations
            if real_frame_id not in annos_per_frame:
                continue
            # frame with empty valid (no ignored region) annotations
            annos_cur_frame = annos_per_frame[real_frame_id]
            if len(annos_cur_frame) == 0:
                continue

            # real_frame_id starts from 0 in KITTI-MOTS, and 1 in MOTSChallenge
            if "kitti" in args["dataset_name"]:
                assert real_frame_id == i
                img_file_path = f"image_02/{seq}/{img_name}"
            else:
                assert real_frame_id == i + 1
                img_file_path = f"{seq}/img1/{img_name}"

            img = cv2.imread(osp.join(imgs_path, img_name))
            img_h, img_w = img.shape[:2]
            frame_id += 1
            img_id = img_cnt + frame_id

            img_info = {
                "file_name": img_file_path,
                "frame_id": frame_id,  # image number in the video sequence, starting from 1.
                "height": img_h,
                "width": img_w,
                "id": img_id,  # image number in the entire dataset, starting from 1.
                # "prev_image_id": img_id - 1 if i > 0 else -1,  # image number in the entire dataset
                # "next_image_id": img_id + 1 if i < num_valid_frames - 1 else -1,
                "video_id": seq_id,
                "num_frames": num_valid_frames
            }
            if num_valid_frames < num_imgs:
                img_info["real_frame_id"] = real_frame_id
            out["images"].append(img_info)

            for anno in annos_cur_frame:
                # unique object id in the entire dataset
                obj_id = f"{seq_id}_{anno['obj_id']}"
                if obj_id not in obj_ids["db"]:
                    obj_ids["db"][obj_id] = obj_ids["cnt"]
                    obj_ids["cnt"] += 1

                cat_id = anno["category_id"]
                cat_name = COCO_CATEGORIES[cat_id]
                cat_inst_ids = inst_ids[cat_name]
                if obj_id not in cat_inst_ids["db"]:
                    cat_inst_ids["db"][obj_id] = cat_inst_ids["cnt"]
                    cat_inst_ids["cnt"] += 1
                    num_insts[cat_name] += 1

                num_annos[cat_name] += 1
                ann_cnt += 1
                anno.update(
                    image_id=img_id,
                    id=ann_cnt,  # annotation id in the entire dataset, starting from 1
                    inst_id=obj_ids["db"][obj_id],  # instance id across all categories in the entire dataset, starting from 0
                    cat_inst_id=cat_inst_ids["db"][obj_id],  # instance id within each category in the entire dataset, starting from 0
                )
                out["annotations"].append(anno) 

        img_cnt += num_valid_frames
        for cat, cnt in num_annos.items():
            annos_cnt[cat] += cnt
        print(f"seq: {seq}, images: {num_valid_frames}/{num_imgs}, size: {img_w}x{img_h}, annos: {num_annos}, instances: {num_insts}")
        # print(obj_ids["cnt"], {cat: insts["cnt"] for cat, insts in inst_ids.items()})

    insts_cnt = {cat: insts["cnt"] for cat, insts in inst_ids.items()}
    print(f"\ndataset:{args['dataset_name']}\nimages: {img_cnt}/{imgs_cnt}\nannos: {annos_cnt}\ninstances: {insts_cnt}")
    with open(args["out_json"], 'w') as f:
        json.dump(out, f, indent=4)
    print(f"COCO-style annotations saved in {args['out_json']}\n")


def cocofy_mots_challenge(seqs_annos, istrain=True, val_seq=""):
    if not istrain and val_seq == "":
        return

    args = {}
    args["seqs_annos"] = seqs_annos
    args["out"] = {
        "info": {"description": "MOTSChallenge dataset."},
        "categories": [
            {"supercategory": "person", "id": 1, "name": "pedestrian"}
        ],
        "images": [],
        # "videos": [],
        "annotations": []
    }

    data_path = osp.join(MOTS_PATH, "train")
    if istrain:
        seqs = sorted([
            seq for seq in os.listdir(data_path)
            if seq.startswith("MOTS") and seq[-2:] != val_seq
        ])
        args["dataset_name"] = f"mots_train_{val_seq}" if val_seq else "mots_train"
    else:
        seqs = [f"MOTS20-{val_seq}"]
        args["dataset_name"] = f"mots_val_{val_seq}"

    args["seqs"] = seqs
    args["imgs_path"] = [osp.join(data_path, f"{seq}/img1") for seq in seqs]
    args["out_json"] = osp.join(ANNO_PATH, f"{args['dataset_name']}.json")

    print(f"\nStarting COCOfying MOTS-Challenge with videos: {seqs}")
    cocofy_mots(args)


def cocofy_kitti_mots(seqs_annos, istrain=True, val_seq_ids=[]):
    if not istrain and len(val_seq_ids) == 0:
        return

    args = {}
    args["seqs_annos"] = seqs_annos
    args["out"] = {
        "info": {"description": "KITTI-MOTS dataset."},
        "categories": [
            {"supercategory": "person", "id": 1, "name": "pedestrian"},
            {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
            {"supercategory": "vehicle", "id": 3, "name": "car"},
        ],
        "images": [],
        "annotations": []
    }

    data_path = osp.join(KITTI_PATH, "training/image_02")
    if istrain:
        seqs = sorted([
            seq for seq in os.listdir(data_path)
            if seq.startswith("00") and int(seq) not in val_seq_ids
        ])
        args["dataset_name"] = "kitti_mots_train" if len(val_seq_ids) > 0 else "kitti_mots_train_full"
    else:
        seqs = [f"{seq:04d}" for seq in val_seq_ids]
        args["dataset_name"] = "kitti_mots_val"

    args["seqs"] = seqs
    args["imgs_path"] = [osp.join(data_path, seq) for seq in seqs]
    args["out_json"] = osp.join(ANNO_PATH, f"{args['dataset_name']}.json")

    print(f"\nStarting COCOfying KITTI-MOTS with videos: {seqs}")
    cocofy_mots(args)


def get_parser():
    parser = argparse.ArgumentParser(description="Convert MOTS to COCO")
    parser.add_argument(
        "--mots", action="store_true", help="cocofy MOTS-Challenge",
    )
    parser.add_argument(
        "--kitti", action="store_true", help="cocofy KITTI-MOTS",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.kitti:
        anno_path = osp.join(ANNO_PATH, "kitti_txt")
        # dict, {seq_id: annos_per_seq}
        seqs_annos = load_seqs_annos_from_txt(anno_path)
        val_seq_ids = [2, 6, 7, 8, 10, 13, 14, 16, 18]
        cocofy_kitti_mots(seqs_annos, istrain=True, val_seq_ids=val_seq_ids)
        cocofy_kitti_mots(seqs_annos, istrain=False, val_seq_ids=val_seq_ids)
        cocofy_kitti_mots(seqs_annos, istrain=True, val_seq_ids=[])

    if args.mots:
        anno_path = osp.join(ANNO_PATH, "mots_txt")
        seqs_annos = load_seqs_annos_from_txt(anno_path)
        for val_seq in ["02", "05", "09", "11"]:
            cocofy_mots_challenge(seqs_annos, istrain=True, val_seq=val_seq)
            cocofy_mots_challenge(seqs_annos, istrain=False, val_seq=val_seq)

        cocofy_mots_challenge(seqs_annos, istrain=True, val_seq="")
