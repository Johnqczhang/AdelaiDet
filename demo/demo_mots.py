"""
Demo for multi-object tracking and segmentation, which takes a video sequence as input,
outputs instance segmentation predictions for each video frame, and associates these predicted
instances across frames for tracking.

Usage: 
    python demo/demo_mots.py --config-file /path/to/cfg.yaml --input [input_args] --output output_dir --opts [cfgs]
        --input [input_args]:
            "kitti val": run demo on all sequences of MOTS-KITTI validation set.
            "kitti test": run demo on all sequences of MOTS-KITTI test set.
            "kitti [val | test]-seq_id": run demo on the specified sequence of MOTS-KITTI dataset.
            "mots val": run demo on all sequences of MOTSChallenge validation set.
            "mots test": run demo on all sequences of MOTSChallenge test set.
            "mots seq_id": run demo on the specified sequence of MOTSChallenge dataset.

    python demo/demo_mots.py --config-file /path/to/cfg.yaml --video-input /path/to/video.mp4 --ouput output_dir --opts [cfgs]

Author: Johnqczhang

"""

import argparse
import multiprocessing as mp
import os
import os.path as osp
import cv2
import time
import torch
import numpy as np
import pycocotools.mask as mask_util

from detectron2.data.detection_utils import read_image

from adet.config import get_cfg
from adet.utils.tracker import MOTSTracker

from demo import setup_cfg
from predictor import VisualizationDemo


COCO_CAT_ID_TO_MOTS = {
    0: 2,  # pedestrian
    2: 1,  # car
}
MOT_PATH = osp.join(osp.dirname(__file__), "../datasets/mot")
DATA_PATH = osp.join(MOT_PATH, "data")


def get_parser():
    parser = argparse.ArgumentParser(description="MOTS Demo")
    parser.add_argument(
        "--config-file",
        default="configs/CondInst/mots_R_50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--with-biou", action="store_true", help="Tracking with bounding box IoU.")
    parser.add_argument("--with-miou", action="store_true", help="Tracking with mask IoU.")
    parser.add_argument("--with-motion", action="store_true", help="Tracking with a motion model.")
    parser.add_argument("--with-embeds", action="store_true", help="Tracking with embeddings.")
    parser.add_argument("--momentums", nargs="+", help="Momentums to update the measurements.")
    parser.add_argument("--match-biou-thr", type=float, default=0.3, help="matching thresh for box IoU")
    parser.add_argument("--match-miou-thr", type=float, default=0.5, help="matching thresh for mask IoU")
    parser.add_argument("--match-reid-thr", type=float, default=0.5, help="matching thresh for embeddings")
    parser.add_argument("--num-tentative", type=int, default=3, help="Number of continuous frames to confirm a track.")
    # parser.add_argument("--num-frames-retain", type=int, default=30, help="Number of continuous frames to remove a track.")
    parser.add_argument("--vis", action="store_true", help="Save visualization results into images.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--with-deduplication", action="store_true", help="Remove duplicate tracks based on box IoU.")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def mask2rles(masks):
    rles = [
        mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        for mask in masks
    ]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")

    return rles


def remove_mask_overlap(masks):
    """ Get non-overlapping masks where each pixel can be assigned to at most one object.

    Args:
        masks (Tensor): NxHxW, binary segmentation masks for N instances.

    Returns:
        new_masks (List[Tensor]): a list of instance masks which are non-overlapping.
    """
    empty_masks = torch.zeros_like(masks)
    mask_unfill = torch.zeros_like(masks[0])
    new_masks = []

    for mask, empty_mask in zip(masks, empty_masks):
        pos = (mask_unfill == 0) & (mask == 1)
        mask_unfill[pos] = 1
        empty_mask[pos] = 1
        new_masks.append(empty_mask)

    return new_masks


def parse_mots_results(frame_id, instances):
    """ Extract txt-format results from predictions for MOTS evaluation. """
    mots_results = []
    if len(instances) == 0:
        return mots_results

    # sort instances according to predicted class score in descending order
    inds = instances.scores.argsort(descending=True)
    instances = instances[inds]
    rles = mask2rles(remove_mask_overlap(instances.pred_masks))
    track_ids = instances.track_ids.numpy()
    cls_ids = instances.pred_classes.numpy()
    img_h, img_w = instances.image_size

    for t_id, c_id, rle in zip(track_ids, cls_ids, rles):
        cat_id = COCO_CAT_ID_TO_MOTS[c_id]
        mots_results.append(f"{frame_id} {t_id} {cat_id} {img_h} {img_w} {rle['counts']}\n")

    return mots_results


def infer_with_tracking(args, demo, tracker):
    if args.vis:
        out_imgs_dir = osp.join(args.output, "images")
        if not osp.exists(args.output) or not osp.exists(out_imgs_dir):
            os.system(f"mkdir -p {out_imgs_dir}")

    infer_time = 0
    txt_results = []
    for frame_id, path in enumerate(args.imgs):  # frame_id: 0-index
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        pred_instances, visualized_output = demo.run_on_image_with_tracker(
            img, tracker, frame_id, args.vis
        )
        runtime = time.time() - start_time
        infer_time += runtime
        print(
            f"{path}: detected {len(pred_instances):2d} instances in {runtime:.2f}s"
        )
        if args.mots_txt:
            img_id = int(path.split('/')[-1].split('.')[0])
            txt_results.extend(parse_mots_results(img_id, pred_instances))
        if args.vis:
            vis_img_filename = osp.join(out_imgs_dir, f"{osp.basename(path).split('.')[0]}.jpg")
            visualized_output.save(vis_img_filename)

    if args.mots_txt:
        with open(args.mots_txt, 'w') as f:
            f.writelines(txt_results)
        print(f"MOTS results saved in {args.mots_txt}, runtime: {infer_time:.2f} s")

    if args.vis:
        video_file = osp.join(args.output, f"{args.output.split('/')[-1]}.mp4")
        video_cmd = (
            f"ffmpeg -threads 2 -y -f image2 -r {args.fps} -i {out_imgs_dir}/%06d.jpg "
            f"-b:v 5000k -c:v mpeg4 {video_file}"
        )
        os.system(video_cmd)
        print(f"Video saved in {video_file}")

    return infer_time


def infer_on_mots(args, demo):
    mots_seqs = [
        osp.join(DATA_PATH, "MOTS-train", seq) for seq in os.listdir(osp.join(DATA_PATH, "MOTS-train"))
        if seq.startswith("MOTS")
    ]
    mots_seqs += [
        osp.join(DATA_PATH, "MOTS-test", seq) for seq in os.listdir(osp.join(DATA_PATH, "MOTS-test"))
        if seq.startswith("MOTS")
    ]
    if args.input[1] == "val":
        mots_seqs = mots_seqs[:4]
    elif args.input[1] == "test":
        mots_seqs = mots_seqs[-4:]
    else:
        mots_seqs = [
            seq for seq in mots_seqs if seq[-2:] in args.input[1:]
        ]
    if len(mots_seqs) == 0:
        raise KeyError(f"Got empty seqs by args.input: {args.input}")

    seqs_name = [seq.split('/')[-1] for seq in mots_seqs]
    print(f"Infer on {seqs_name} ...")
    out_dir = args.output
    if not osp.exists(out_dir):
        os.system(f"mkdir -p {out_dir}")

    mots_fps = {
        "MOTS20-02": 30,
        "MOTS20-05": 14,
        "MOTS20-09": 30,
        "MOTS20-11": 30,
        "MOTS20-01": 30,
        "MOTS20-06": 14,
        "MOTS20-07": 30,
        "MOTS20-12": 30,
    }

    tracker = MOTSTracker(args)
    total_time = 0
    for seq_name, seq_path in zip(seqs_name, mots_seqs):
        img_dir = osp.join(seq_path, "img1")
        args.imgs = sorted([
            osp.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")
        ])
        args.fps = mots_fps[seq_name]
        if seq_name[-2:] in ["01", "06", "07", "12"]:
            args.mots_txt = osp.join(out_dir, f"{seq_name}.txt")
        else:
            args.mots_txt = osp.join(out_dir, f"00{seq_name[-2:]}.txt")
        args.output = osp.join(out_dir, seq_name)

        print(f"Inferring on {seq_name} ...")
        tracker.reset(args.fps)
        infer_time = infer_with_tracking(args, demo, tracker)
        total_time += infer_time

    print(f"Total time: {total_time:.2f} s")


def infer_on_kitti_mots(args, demo):
    data_path = osp.join(DATA_PATH, "kitti")
    inputs = args.input[1].split('-')
    assert inputs[0] in ["val", "test"]

    if inputs[0] == "val":
        seqs_id = [2, 6, 7, 8, 10, 13, 14, 16, 18]
        if len(inputs) > 1:
            assert int(inputs[1]) in seqs_id
            seqs_id = [int(inputs[1])]
        seqs = [
            osp.join(data_path, "training/image_02", seq)
            for seq in os.listdir(osp.join(data_path, "training/image_02"))
            if int(seq) in seqs_id
        ]
    else:
        seqs = [
            osp.join(data_path, "testing/image_02", seq)
            for seq in os.listdir(osp.join(data_path, "testing/image_02"))
        ]

    seqs_name = [seq.split('/')[-1] for seq in seqs]
    args.fps = 10
    print(f"Infer on {seqs_name} ...")
    out_dir = args.output
    if not osp.exists(out_dir):
        os.system(f"mkdir -p {out_dir}")

    trackers = {
        k : MOTSTracker(args, frame_rate=args.fps) for k in ["ped", "car"]
    }

    total_time = 0
    for seq_name, seq_path in zip(seqs_name, seqs):
        args.imgs = sorted([
            osp.join(seq_path, f) for f in os.listdir(seq_path) if f.endswith(".png")
        ])
        args.mots_txt = osp.join(out_dir, f"{seq_name}.txt")
        args.output = osp.join(out_dir, f"kitti-{seq_name[-2:]}")
        print(f"Inferring on {seq_name} ...")
        infer_time = infer_with_tracking(args, demo, trackers)
        total_time += infer_time

        for _, tracker in trackers.items():
            tracker.reset(args.fps)

    print(f"Total time: {total_time:.2f} s")


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    print("Arguments: " + str(args))
    cfg = setup_cfg(args)

    if args.momentums and len(args.momentums) > 0:
        assert len(args.momentums) % 2 == 0
        momentums = {
            args.momentums[i]: args.momentums[i + 1]
            for i in range(0, len(args.momentums), 2)
        }
        args.momentums = momentums

    demo = VisualizationDemo(cfg)

    if args.input:
        if args.input[0] == "mots":
            infer_on_mots(args, demo)
        elif args.input[0] == "kitti":
            infer_on_kitti_mots(args, demo)
    elif args.video_input:
        assert osp.isfile(args.video_input)
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        assert args.output
        out_imgs_dir = osp.join(args.output, "images")
        if not osp.exists(args.output) or not osp.exists(out_imgs_dir):
            os.system(f"mkdir -p {out_imgs_dir}")

        trackers = {
            k : MOTSTracker(args, frame_rate=fps) for k in ["ped", "car"]
        }
        frame_gen = _frame_from_video(video)

        infer_time = 0
        for frame_id, frame in enumerate(frame_gen):
            start_time = time.time()
            pred_instances, visualized_output = demo.run_on_image_with_tracker(
                frame, trackers, frame_id, True
            )
            runtime = time.time() - start_time
            infer_time += runtime
            print(
                f"frame-{frame_id}: detected {len(pred_instances):2d} instances in {runtime:.2f}s"
            )
            vis_img_filename = osp.join(out_imgs_dir, f"{frame_id:06d}.jpg")
            visualized_output.save(vis_img_filename)

        video_file = osp.join(args.output, f"{basename.split('.')[0]}_mots.mp4")
        video_cmd = (
            f"ffmpeg -threads 2 -y -f image2 -r {fps} -i {out_imgs_dir}/%06d.jpg "
            f"-b:v 5000k -c:v mpeg4 {video_file}"
        )
        os.system(video_cmd)
        print(f"Video saved in {video_file}, runtime: {infer_time:.2f} s")
    else:
        raise NotImplementedError
