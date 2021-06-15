import numpy as np
import torch
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

    def compute_dists_by_miou(self, cur_ids, track_ids):
        cur_masks = mask2rles(self.instances.pred_masks[cur_ids])
        track_masks = mask2rles(self.get("metrics", track_ids))
        iscrowd = [0 for _ in range(len(cur_masks))]
        mious = mask_util.iou(track_masks, cur_masks, iscrowd)
        return 1 - mious

    def get(self, item, ids, num_samples=None, behavior=None):
        if self.track_metric == "miou":
            return super().get(item, ids)
        elif self.track_metric == "pixel_embeds":
            outs = [
                self.tracks[id][item] for id in ids
            ]
            outs = [o[-1].squeeze(0) if isinstance(o, list) else o for o in outs]
            return outs

    def compute_dists_by_embeds(self, cur_ids, track_ids):
        cur_embeds = self.instances.pixel_embeds[cur_ids].embeds_list
        track_embeds = self.get("metrics", track_ids)
        m, n = len(track_embeds), len(cur_embeds)
        dists = torch.zeros((m, n), dtype=torch.float32)
        for i in range(m):
            p1 = track_embeds[i]
            # m1 = torch.linalg.norm(p1, dim=1)
            for j in range(n):
                p2 = cur_embeds[j]
                # m2 = torch.linalg.norm(p2, dim=1)
                # mod = (m1[:, None] * m2[None]).clip(min=1e-8)
                # dists[i, j] = 1 - (p1.matmul(p2.t()) / mod).mean()
                dists = (p1[:, None] - p2[None]).square().sum(dim=-1).sqrt()
        return dists.numpy()

    def track(self, frame_id, **kwargs):
        num_inst = len(self.instances)
        if num_inst == 0:
            return torch.arange(num_inst)

        if self.track_metric == "miou":
            assert self.instances.has("pred_masks")
            metrics = self.instances.pred_masks
        elif self.track_metric == "pixel_embeds":
            assert self.instances.has("pixel_embeds")
            metrics = self.instances.pixel_embeds

        if self.empty or num_inst == 0:
            ids = torch.arange(self.num_tracks, self.num_tracks + num_inst)
            self.num_tracks += num_inst
        else:
            ids = torch.full((num_inst,), -1, dtype=torch.long)
            track_ids = [
                id for id in self.ids if id not in ids
                and self.tracks[id].frame_ids[-1] == frame_id - 1
            ]
            if len(track_ids) > 0:
                cur_ids = torch.nonzero(ids == -1).squeeze(1)
                if self.track_metric == "miou":
                    dists = self.compute_dists_by_miou(cur_ids, track_ids)
                elif self.track_metric == "pixel_embeds":
                    dists = self.compute_dists_by_embeds(cur_ids, track_ids)  
                row, col = linear_sum_assignment(dists)
                for r, c in zip(row, col):
                    if dists[r, c] < 1 - self.match_thr:
                        ids[cur_ids[c]] = track_ids[r]

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks, self.num_tracks + new_track_inds.sum()
            )
            self.num_tracks += new_track_inds.sum()

        self.update(ids=ids, frame_ids=frame_id, metrics=metrics)
        return ids + 1


def parse_mots_results(img, frame_id, instances, track_ids):
    mots_results = ""
    num_inst = len(instances)
    if num_inst == 0:
        return mots_results

    img_h, img_w = img.shape[:2]
    rles = remove_mask_overlap(instances)
    cls_id = 2

    for track_id, rle in zip(track_ids, rles):
        obj_id = 2000 + track_id
        mots_results += f"{frame_id} {obj_id} {cls_id} {img_h} {img_w} {rle['counts']}\n"

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
