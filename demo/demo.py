# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import os.path as osp
import time
import cv2
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

from mask_tracker import MaskTracker
from mask_tracker import parse_mots_results


# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--track", action="store_true", help="Tracking by mask IoU.")
    parser.add_argument("--mots-seqs", nargs="+", help="A list of space separated mots videos ids")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
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


def infer_with_tracking(args, demo, tracker):
    out_imgs_dir = osp.join(args.output, "images")
    if not osp.exists(args.output) or not osp.exists(out_imgs_dir):
        os.system(f"mkdir -p {out_imgs_dir}")

    if args.mots_txt and osp.exists(args.mots_txt):
        os.system(f"rm -rf {args.mots_txt}")

    infer_time = 0
    for frame_id, path in enumerate(tqdm.tqdm(args.imgs, disable=args.mots_txt), 1):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output, track_ids = demo.run_on_image_with_tracker(
            img, tracker, frame_id
        )
        runtime = time.time() - start_time
        infer_time += runtime
        print(
            f"{path}: detected {len(predictions['instances'])} instances in {runtime:.2f}s"
        )
        if args.mots_txt:
            mots_results = parse_mots_results(img, frame_id, tracker.instances, track_ids)
            with open(args.mots_txt, 'a+') as f:
                f.writelines(mots_results)
        out_filename = osp.join(out_imgs_dir, osp.basename(path))
        visualized_output.save(out_filename)

    video_cmd = f"ffmpeg -threads 2 -y -f image2 -r {args.fps} -i {out_imgs_dir}/%06d.jpg"
    video_file = osp.join(args.output, f"{args.output.split('/')[-1]}.mp4")
    video_cmd += f" -b:v 5000k -c:v mpeg4 {video_file}"
    os.system(video_cmd)

    return infer_time


def infer_on_mots(args, demo):
    mots_path = osp.join(osp.dirname(__file__), "../datasets/mots/MOTS")
    mots_seqs = [
        osp.join(mots_path, "train", seq) for seq in os.listdir(osp.join(mots_path, "train"))
        if seq.startswith("MOTS")
    ]
    mots_seqs += [
        osp.join(mots_path, "test", seq) for seq in os.listdir(osp.join(mots_path, "test"))
        if seq.startswith("MOTS")
    ]
    if args.input[1] == "train":
        mots_seqs = mots_seqs[:4]
        out_dir = osp.join(args.output, "train")
    elif args.input[1] == "test":
        mots_seqs = mots_seqs[-4:]
        out_dir = osp.join(args.output, "test")
    else:
        mots_seqs = [
            seq for seq in mots_seqs if seq[-2:] in args.input[1:]
        ]
        out_dir = osp.join(args.output, "mix")
    seqs_name = [seq.split('/')[-1] for seq in mots_seqs]
    print(f"Infer on {seqs_name} ...")
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

    tracker = MaskTracker()
    total_time = 0
    for seq_name, seq_path in zip(seqs_name, mots_seqs):
        img_dir = osp.join(seq_path, "img1")
        args.imgs = [osp.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]
        args.imgs = sorted(args.imgs)
        args.fps = mots_fps[seq_name]
        args.output = osp.join(out_dir, seq_name)
        args.mots_txt = osp.join(args.output, f"{seq_name}.txt")
        print(f"Inferring on {seq_name} ...")
        infer_time = infer_with_tracking(args, demo, tracker)
        print(f"Results saved in {args.output}, runtime: {infer_time:.2f} s")
        total_time += infer_time

        tracker.reset()

    print(f"Total time: {total_time:.2f} s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input[0] == "mots":
       infer_on_mots(args, demo)
    elif args.track:
        args.imgs = [osp.join(args.input[0], fname) for fname in os.listdir(args.input[0]) if fname.endswith(".jpg")]
        args.imgs = sorted(args.imgs)
        args.fps = 30
        tracker = MaskTracker()
        infer_time = infer_with_tracking(args, demo, tracker)
        print(f"Inference time: {infer_time:.2f} s")
    elif args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
