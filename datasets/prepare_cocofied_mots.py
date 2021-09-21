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
import copy

from pycocotools import mask as maskUtils
# # USER: Debug for visualization of annotations
# import matplotlib as mpl
# import matplotlib.colors as mplc
# import matplotlib.pyplot as plt
# from detectron2.utils.colormap import random_color
# from adet.data.dataset_mapper import segmToMask


MOT_PATH = osp.join(osp.dirname(__file__), "mot")
KITTI_PATH = osp.join(MOT_PATH, "kitti")
MOTS_PATH = osp.join(MOT_PATH, 'mots')
ANNO_PATH = osp.join(MOT_PATH, 'annotations')
if not osp.exists(ANNO_PATH):
    os.makedirs(ANNO_PATH)

MOTS_CATEGORIES = {1: "car", 2: "pedestrian", 10: "ignored"}


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
            cat_cnt[10] += 1
            continue

        assert obj_id not in obj_ids_per_frame[frame], f"Multiple objects with the same id: {obj_id} in frame-{frame}"
        obj_ids_per_frame[frame].add(obj_id)

        cat_id = int(fields[2])
        # cat_id should be either 1 for "car" or 2 for "pedestrian"
        assert cat_id in MOTS_CATEGORIES, f"Unknown object class id: {cat_id}"
        cat_cnt[cat_id] += 1

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
            "category_id": 1 if cat_id == 2 else 2,  # use unified category id, pedestrian: 1, car: 2
            "obj_id": obj_id,
            "bbox": maskUtils.toBbox(mask).tolist(),  # enclosing bbox, fmt: xywh
            "segmentation": mask,
            "area": float(area),
            "iscrowd": 0
        })

    cat_cnt = {MOTS_CATEGORIES[k]: v for k, v in cat_cnt.items() if v > 0}
    print(f"Number of annos: {cat_cnt}")
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


def get_cocofied_json_data(args):
    out = {
        "info": {"description": f"{args['dataset_name']}."},
        "categories": args["categories"],
        # "videos": [],
        "images": [],
        "annotations": []
    }

    # count the number of frames in the dataset
    imgs_cnt = {"valid": 0, "total": 0}
    # all category names in the dataset
    cat_names = {cat["id"]: cat["name"] for cat in args["categories"]}
    # count the number of annotated objects per category in the dataset
    annos_cnt = {c: 0 for c in cat_names}
    # count the number of annotated indentities per category in the dataset
    objs_dict = {c: {} for c in cat_names}
    # count the width, height, aspect ration, and area of boxes in each category
    boxes_cnt = {c: [] for c in cat_names}

    print(f"\nStart COCOfying {args['dataset_name']} with videos: {args['seqs']}")

    for seq, imgs_path in zip(args["seqs"], args["imgs_path"]):
        seq_id = int(seq[-2:])
        # dict, {frame_id: [{obj1_annos}, {obj2_annos}, ...]}
        seq_annos = args["seqs_annos"][seq_id]

        img_files = sorted([
            img for img in os.listdir(imgs_path) if img.endswith(args["imExt"])
        ])
        num_imgs = len(img_files)

        # debug_path = osp.join(imgs_path, "../debug")
        # if not osp.exists(debug_path):
        #     os.mkdir(debug_path)
        # args["debug_path"] = debug_path

        # count the number of annotated objects per category in the current sequence.
        seq_annos_cnt = {c: 0 for c in cat_names}
        # count the number of annotated identities per category in the current sequence.
        seq_objs_cnt = {c: 0 for c in cat_names}
        # count the number of frames which contain at least one annotation of an object
        img_cnt = 0
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
                # "prev_image_id": img_id - 1 if i > 0 else -1,  # image number in the entire dataset
                # "next_image_id": img_id + 1 if i < num_valid_imgs - 1 else -1,
                "seq_id": seq_id,
                "num_frames": num_imgs
            }
            out["images"].append(img_info)
            img_cnt += 1

            # boxes, masks = [], []

            for obj in annos_cur_frame:
                cat_id = obj["category_id"]

                if cat_id not in cat_names:
                    # boxes.append(np.array(obj["bbox"] + [cat_id]))
                    # masks.append(segmToMask(obj["segmentation"], (img_h, img_w)))
                    continue

                # hard copy of each object's annotation to avoid reference
                anno = copy.deepcopy(obj)
                # unique object identity id in the entire dataset: seqId_catId_objId
                obj_uId = f"{args['dataset_name']}_{seq_id}_{cat_id}_{anno['obj_id']}"

                # encounter a new object identity in the dataset
                if obj_uId not in objs_dict[cat_id]:
                    # assign a unique identity id in the entire dataset, starting from 0.
                    obj_id = sum([len(v) for v in objs_dict.values()])
                    objs_dict[cat_id][obj_uId] = obj_id
                    seq_objs_cnt[cat_id] += 1
                else:
                    obj_id = objs_dict[cat_id][obj_uId]

                # annotation id in the entire dataset, starting from 0.
                anno_id = sum([v for v in annos_cnt.values()])
                anno.update(
                    image_id=img_id,
                    id=anno_id,
                    obj_id=obj_id,
                    seq_id=seq_id,
                )
                out["annotations"].append(anno)
                annos_cnt[cat_id] += 1
                seq_annos_cnt[cat_id] += 1

                box_w, box_h = anno["bbox"][2:4]
                boxes_cnt[cat_id].append([box_w, box_h, box_w / box_h, box_w * box_h])

            # if len(boxes) == 0:
            #     continue
            # if seq_id == 1:
            #     continue
            # dpi = 200
            # fig = plt.figure(figsize=[img_w / dpi, img_h / dpi], frameon=False)
            # ax = plt.Axes(fig, [0., 0., 1., 1.]);  ax.axis("off");  fig.add_axes(ax)
            # ax.imshow(img[..., ::-1])
            # boxes = np.array(boxes)
            # for j, box in enumerate(boxes):
            #     x0, y0, w, h, cat_id = box
            #     color = random_color(rgb=True, maximum=1)
            #     ax.add_patch(
            #         mpl.patches.Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=2, alpha=0.8, linestyle='-')
            #     )
            #     # ax.text(
            #     #     x0, y0, f"{cat_id}", color=color
            #     # )
            #     rgba = np.zeros((img_h, img_w, 4), dtype=np.float32)
            #     rgba[..., :3] = mplc.to_rgb(color)
            #     rgba[..., 3] = (masks[j] == 1).astype(np.float32) * 0.5
            #     ax.imshow(rgba)

            # plt.savefig(osp.join(args["debug_path"], filename), dpi=dpi)
            # plt.close("all")

        imgs_cnt["valid"] += img_cnt
        imgs_cnt["total"] += num_imgs
        seq_annos_cnt = {cat_names[c]: v for c, v in seq_annos_cnt.items()}
        seq_objs_cnt = {cat_names[c]: v for c, v in seq_objs_cnt.items()}
        print(f"seq: {seq}, images: {img_cnt}/{num_imgs}, size: {img_w}x{img_h}, annos: {seq_annos_cnt}, objs: {seq_objs_cnt}")

    annos_cnt = {cat_names[c]: v for c, v in annos_cnt.items()}
    objs_cnt = {cat_names[c]: len(v) for c, v in objs_dict.items()}
    print(f"\ndataset: {args['dataset_name']}\nimages: {imgs_cnt['valid']}/{imgs_cnt['total']}\nannos: {annos_cnt}\nobjs: {objs_cnt}")
    for c in boxes_cnt:
        boxes = np.array(boxes_cnt[c])
        print(f"{cat_names[c]} (whas): min: {boxes.min(axis=0)}, max: {boxes.max(axis=0)}, mean: {boxes.mean(axis=0)}")
    print()

    return out


def cocofy_mots_challenge():
    anno_path = osp.join(ANNO_PATH, "mots_txt")
    data_path = osp.join(MOTS_PATH, "train")

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

    json_file = osp.join(ANNO_PATH, "mots_train_full.json")
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
    anno_path = osp.join(ANNO_PATH, "kitti_txt")
    data_path = osp.join(KITTI_PATH, "training/image_02")

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
        {"id": 2, "name": "car", "supercategory": "vehicle"},
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
        cocofy_kitti_mots()
    if args.mots:
        cocofy_mots_challenge()
