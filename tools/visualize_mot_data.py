import argparse
import os
import os.path as osp
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from detectron2.utils.colormap import random_color


MOT_PATH = osp.join(osp.dirname(__file__), "../datasets/mot")
ANNO_PATH = osp.join(MOT_PATH, "annotations")
DATA_PATH = osp.join(MOT_PATH, "data")


def get_parser():
    parser = argparse.ArgumentParser(description="Visualize MOT/MOTS data with annotations/detections")
    parser.add_argument(
        "--mots", action="store_true", help="MOTS-Challenge",
    )
    parser.add_argument(
        "--kitti-mots", action="store_true", help="KITTI-MOTS",
    )
    parser.add_argument(
        "--ht21", action="store_true", help="Crowd of Heads Dataset (CrpHD) for Head Tracking",
    )

    return parser


def load_mot_det_txt(txt_path):
    boxes_per_frame = {}
    print(f"Loading detections from {txt_path}")
    dets = np.loadtxt(txt_path, dtype=np.float32, delimiter=',')
    for fields in dets:
        frame_id = int(fields[0])
        if frame_id not in boxes_per_frame:
            boxes_per_frame[frame_id] = []
        
        box = [float(x) for x in fields[2:6]]  # xywh
        boxes_per_frame[frame_id].append(box)
    
    return boxes_per_frame


def vis_ht21_test():
    data_path = osp.join(DATA_PATH, "HT21-test")
    seqs = [seq for seq in os.listdir(data_path) if seq.startswith("HT21")]

    for seq in seqs:
        seq_path = osp.join(data_path, seq)
        imgs_path = osp.join(seq_path, "img1")
        boxes_per_frame = load_mot_det_txt(osp.join(seq_path, "det/det.txt"))
        out_dir = osp.join(seq_path, "det/imgs")
        if not osp.exists(out_dir):
            os.mkdir(out_dir)

        img_files = sorted([
            img for img in os.listdir(imgs_path) if img.endswith(".jpg")
        ])
        for filename in img_files:
            frame_id = int(filename.split('.')[0])
            if frame_id not in boxes_per_frame:
                continue
            if len(boxes_per_frame[frame_id]) == 0:
                continue

            img = cv2.imread(osp.join(imgs_path, filename))
            img_h, img_w = img.shape[:2]
            boxes = boxes_per_frame[frame_id]
            print(f"Detect {len(boxes):3d} boxes in {seq}/{filename}")

            dpi = 200
            fig = plt.figure(figsize=[img_w / dpi, img_h / dpi], frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.]);  ax.axis("off");  fig.add_axes(ax)
            ax.imshow(img[..., ::-1])
            for box in boxes:
                x0, y0, w, h = box
                color = random_color(rgb=True, maximum=1)
                ax.add_patch(
                    mpl.patches.Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=1, alpha=0.8, linestyle='-')
                )
            plt.savefig(osp.join(out_dir, filename), dpi=dpi)
            plt.close("all")


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.ht21:
        vis_ht21_test()
    else:
        raise NotImplementedError
