# -*- coding: utf-8 -*-
"""
based on https://github.com/PeizeSun/TransTrack/blob/main/track_tools/convert_mot_to_coco.py
MOTS dataset: https://motchallenge.net/data/MOTS.zip
Modified by Johnqczhang
"""
import os
import os.path as osp
import argparse
import cv2
import numpy as np
import json

from pycocotools import mask as maskUtils


MOTS_PATH = osp.join(osp.dirname(__file__), 'mots')
ANNO_PATH = osp.join(MOTS_PATH, 'annotations')
DATA_PATH = osp.join(MOTS_PATH, 'train')
if not osp.exists(ANNO_PATH):
    os.makedirs(ANNO_PATH)

MOTS_CATEGORIES = {1: "car", 2: "pedestrian"}


def load_annos_from_txt(txt_path):
    """
    Load annotations from the txt file of a video sequence

    Returns:
        objects_per_frame (dict): {frame_id: list(object_dict)}
    """
    objects_per_frame = {}
    obj_ids_per_frame = {}  # To check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # To check that no frame contains overlapping masks

    annos = np.loadtxt(txt_path, dtype=np.str_, delimiter=' ')
    for fields in annos:
        frame = int(fields[0])
        if frame not in objects_per_frame:
            objects_per_frame[frame] = []
        if frame not in obj_ids_per_frame:
            obj_ids_per_frame[frame] = set()
        
        obj_id = int(fields[1])
        assert obj_id not in obj_ids_per_frame[frame], f"Multiple objects with the same id: {obj_id} in frame-{frame}"
        obj_ids_per_frame[frame].add(obj_id)

        cat_id = int(fields[2])
        if cat_id == 10:  # ignored region
            continue
        assert cat_id in MOTS_CATEGORIES.keys(), f"Unknown object class id: {cat_id}"

        mask = {
            "size": [int(fields[3]), int(fields[4])],  # img_h, img_w
            "counts": str(fields[5])
        }  # binary mask in RLE format 
        if frame not in combined_mask_per_frame:
            combined_mask_per_frame[frame] = mask
        else:
            overlap = maskUtils.area(maskUtils.merge([combined_mask_per_frame[frame], mask], intersect=True))
            assert overlap <= 0., f"Objects with overlapping masks in frame-{frame}"
            combined_mask_per_frame[frame] = maskUtils.merge([combined_mask_per_frame[frame], mask], intersect=False)
        
        bbox = maskUtils.toBbox(mask)  # enclosing bbox, fmt: xywh
        area = maskUtils.area(mask)  # mask area

        objects_per_frame[frame].append(
            {
                # use the category_id in COCO annotation
                "category_id": 3 if MOTS_CATEGORIES[cat_id] == "car" else 1,
                "obj_id": obj_id,
                "bbox": bbox.tolist(),
                "segmentation": mask,
                "area": float(area),
                "iscrowd": 0
            }
        )

    return objects_per_frame


def cocofy_mots(phase, val_seq_id):
    out = {
        "info": {"description": "MOTSChallenge dataset."},
        "categories": [
            {"supercategory": "none", "id": 1, "name": "pedestrian"}
        ],
        "images": [],
        # "videos": [],
        "annotations": []
    }

    data_path = osp.join(MOTS_PATH, "train")
    seqs = [seq for seq in os.listdir(data_path) if seq.startswith("MOTS")]
    if phase == "train":
        if val_seq_id:
            seqs = [seq for seq in seqs if not seq.endswith(val_seq_id)]
            out_json = osp.join(ANNO_PATH, f"train-{val_seq_id}.json")
        else:
            out_json = osp.join(ANNO_PATH, "train.json")
    elif phase == "val":
        seqs = [seq for seq in seqs if seq.endswith(val_seq_id)]
        assert len(seqs) > 0, f"Error: video-{val_seq_id} not found in {data_path}"
        out_json = osp.join(ANNO_PATH, f"val-{val_seq_id}.json")
    print(f"Starting COCOfying MOTS with videos: {seqs}")

    img_cnt = 0
    ann_id = 0
    inst_id = 0  # instance id across the entire data set, starting from 1
    inst_ids_dict = dict()
    for seq in sorted(seqs):
        seq_id = int(seq[-2:])
        # out["videos"].append(
        #     {"id": seq_id, "file_name": seq}
        # )
        seq_dir = osp.join(data_path, seq)
        img_path = osp.join(seq_dir, "img1")
        imgs = [img for img in os.listdir(img_path) if img.endswith(".jpg")]
        num_imgs = len(imgs)
        anno_txt = osp.join(seq_dir, "gt/gt.txt")
        annos_per_frame = load_annos_from_txt(anno_txt)
        num_annos = 0
        num_insts = 0

        for i in range(num_imgs):
            img_file = f"{seq}/img1/{i + 1:06d}.jpg"
            img = cv2.imread(osp.join(data_path, img_file))
            img_h, img_w = img.shape[:2]
            img_id = img_cnt + i + 1
            out["images"].append(
                {
                    "file_name": img_file,
                    "frame_id": i + 1,  # image number in the video sequence, starting from 1.
                    "height": img_h,
                    "width": img_w,
                    "id": img_id,  # image number in the entire data set.
                    "prev_image_id": img_id - 1 if i > 0 else -1,
                    "next_image_id": img_id + 1 if i < num_imgs - 1 else -1,
                    "video_id": seq_id,
                    "num_frames": num_imgs
                }
            )

            annos_cur_frame = annos_per_frame[i + 1]
            num_annos += len(annos_cur_frame)
            for anno in annos_cur_frame:
                obj_id = f"{seq_id}_{anno['obj_id'] % 1000}"
                if obj_id not in inst_ids_dict:
                    inst_id += 1
                    num_insts += 1
                    inst_ids_dict[obj_id] = inst_id
                anno.update(image_id=img_id, id=ann_id, inst_id=inst_ids_dict[obj_id])
                ann_id += 1
                out["annotations"].append(anno) 

        img_cnt += num_imgs
        print(f"{seq}: {num_imgs} images, {num_annos} annotations for {num_insts} instance identities")

    num_insts = len(inst_ids_dict)
    # out["num_insts"] = num_insts
    print(f"{phase}: {len(out['images'])} images, {len(out['annotations'])} annotations for {num_insts} instance identities")
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=4)
    print(f"COCO-style annotations saved in {out_json}")


def get_parser():
    parser = argparse.ArgumentParser(description="Convert MOTS to COCO")
    parser.add_argument(
        "--val-seq",
        default="",
        help="the video sequence number used as cross validation",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cocofy_mots("train", args.val_seq)
    if args.val_seq:
        cocofy_mots("val", args.val_seq)
