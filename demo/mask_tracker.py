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


@TRACKERS.register_module()
class MaskTracker(BaseTracker):
    def __init__(self, track_metric="miou", match_thr=0.7, **kwargs):
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
                id for id in self.ids if id not in ids
                and self.tracks[id].frame_ids[-1] == frame_id - 1
                # and self.tracks[id].frame_ids[-1] >= frame_id - self.num_frames_retain
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


def parse_mots_results(img, frame_id, instances, track_ids):
    mots_results = []
    if len(instances) == 0:
        return mots_results

    img_h, img_w = img.shape[:2]
    rles = remove_mask_overlap(instances)
    cls_id = 2

    for track_id, rle in zip(track_ids, rles):
        obj_id = 2000 + track_id
        mots_results.append(f"{frame_id} {obj_id} {cls_id} {img_h} {img_w} {rle['counts']}\n")

    return mots_results


def remove_mask_overlap(instances):
    scores = instances.scores.numpy()
    sorted_idxs = np.argsort(-scores).tolist()
    masks = [instances.pred_masks[idx] for idx in sorted_idxs]
    mask_unfill = torch.zeros_like(masks[0])
    masks_wo_overlap = []

    for mask in masks:
        new_mask = torch.zeros_like(mask)
        pos = (mask_unfill == 0) & (mask == 1)
        mask_unfill[pos] = 1
        new_mask[pos] = 1
        masks_wo_overlap.append(new_mask)

    rles = mask2rles(masks_wo_overlap)
    return rles
