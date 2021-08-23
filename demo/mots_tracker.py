import numpy as np
import torch
import torch.nn.functional as F
import pycocotools.mask as mask_util

from motmetrics.lap import linear_sum_assignment
from mmtrack.models import TRACKERS
from mmtrack.models.mot.trackers.base_tracker import BaseTracker


def mask2rles(masks):
    rles = [
        mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        for mask in masks
    ]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")

    return rles


def xyxy2xyah(boxes):
    """ Convert bounding boxes from xyxy to xyah.

    Args:
        boxes (Tensor[float]): a Nx4 matrix. Each row is (x1, y1, x2, y2).
    """
    center = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    ah = boxes[:, 2:4] - boxes[:, 0:2]
    ah[:, 0] = ah[:, 0] / ah[:, 1]
    xyah = torch.stack([center, ah], dim=1)
    return xyah


@TRACKERS.register_module()
class MOTSTracker(BaseTracker):
    """ A simple tracker for MOTS.
    The detections and segmentations of all instances in each frame are predicted by a CondInst model.

    Args:
        track_metric (str, optional): the metric used for association. Default: 'miou', i.e., mask IoU.
        match_thr (float, optional): Threshold for whether accept a match. Defaults to 0.5.
    """

    def __init__(
        self,
        track_metric="miou",
        match_thr=0.7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.track_metric = track_metric
        self.match_thr = match_thr

    def compute_dists_by_miou(self, metrics, track_ids):
        track_masks = mask2rles(self.get("masks", track_ids))
        cur_masks = mask2rles(metrics["masks"])
        iscrowd = [0] * len(cur_masks)
        mious = mask_util.iou(track_masks, cur_masks, iscrowd)
        return 1 - mious

    def compute_dists_by_mfeats(self, metrics, track_ids):
        track_mfeats = self.get("mfeats_l2_norm", track_ids)
        cur_mfeats = metrics["mfeats_l2_norm"]
        # For two l2-normalized vectors X, Y, we have
        # ||X - Y||^2 = 2 * (1 - cos(X, Y))
        dists = (track_mfeats[:, None] - cur_mfeats[None]).square().sum(dim=-1) * 0.5
        dists = dists.numpy()

        if "miou" in self.track_metric:
            mious = self.compute_dists_by_miou(metrics, track_ids)
            if "+" in self.track_metric:
                dists = (dists + mious) * 0.5
            elif "*" in self.track_metric:
                dists = 1 - (1 - dists) * (1 - mious)

        return dists

    def track(self, frame_id, instances):
        num_inst = len(instances)
        if num_inst == 0:
            return torch.arange(num_inst)

        metrics = {}

        if "miou" in self.track_metric:  # "miou" means association using mask IoU
            assert instances.has("pred_masks")
            metrics["masks"] = instances.pred_masks
        if "mfeats" in self.track_metric:
            assert instances.has("pred_mfeats")
            # l2 normalization along the feature dimension
            mfeats_norm = F.normalize(instances.pred_mfeats, dim=1)
            metrics["mfeats_l2_norm"] = mfeats_norm  # (num_inst, C)

        if self.empty:
            ids = torch.arange(self.num_tracks, self.num_tracks + num_inst)
            self.num_tracks += num_inst
        else:
            ids = torch.full((num_inst,), -1, dtype=torch.long)
            track_ids = [
                id for id, track in self.tracks.items()
                if id not in ids and track.frame_ids[-1] == frame_id - 1
                # and track.frame_ids[-1] >= frame_id - self.num_frames_retain
            ]
            if len(track_ids) > 0:
                if self.track_metric == "miou":
                    dists = self.compute_dists_by_miou(metrics, track_ids)
                elif "mfeats" in self.track_metric:
                    dists = self.compute_dists_by_mfeats(metrics, track_ids)
                else:
                    raise NotImplementedError

                row, col = linear_sum_assignment(dists)
                for r, c in zip(row, col):
                    if dists[r, c] < 1 - self.match_thr:
                        ids[c] = track_ids[r]

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks, self.num_tracks + new_track_inds.sum()
            )
            self.num_tracks += new_track_inds.sum()

        self.update(ids=ids, frame_ids=frame_id, **metrics)
        return ids + 1


COCO_CAT_ID_TO_MOTS = {
    0: 2,  # pedestrian
    2: 1,  # car
}

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
