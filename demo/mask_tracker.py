import numpy as np
import torch
import pycocotools.mask as mask_util

# from detectron2.structures import BoxMode
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
    def __init__(self, match_iou_thr=0.7, num_tentatives=3, **kwargs):
        super().__init__(**kwargs)
        self.match_iou_thr = match_iou_thr
        self.num_tentatives = num_tentatives

    def track(self, frame_id, **kwargs):
        instances = self.instances
        num_inst = len(instances)
        if num_inst == 0:
            return torch.tensor([], dtype=torch.long)

        # bboxes = instances.pred_boxes.tensor  # fmt: xyxy
        # bboxes = BoxMode.convert(bboxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        # scores = instances.scores
        # labels = torch.full((num_inst, ), 1, dtype=torch.long)
        assert instances.has("pred_masks")
        masks = instances.pred_masks

        if self.empty:
            num_new_tracks = num_inst
            ids = torch.arange(self.num_tracks, self.num_tracks + num_new_tracks, dtype=torch.long)
            self.num_tracks += num_new_tracks
        else:
            ids = torch.full((num_inst,), -1, dtype=torch.long)

            active_ids = [
                id for id in self.ids if id not in ids
                and self.tracks[id].frame_ids[-1] == frame_id - 1
            ]
            if len(active_ids) > 0:
                active_idx = torch.nonzero(ids == -1).squeeze(1)
                track_masks = mask2rles(self.get("masks", active_ids))
                cur_masks = mask2rles(masks[active_idx])
                iscrowd = [0 for _ in range(len(cur_masks))]
                ious = mask_util.iou(track_masks, cur_masks, iscrowd)
                dists = 1 - ious
                row, col = linear_sum_assignment(dists)
                for r, c in zip(row, col):
                    dist = dists[r, c]
                    if dist < 1 - self.match_iou_thr:
                        ids[active_idx[c]] = active_ids[r]

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long)
            self.num_tracks += new_track_inds.sum()

        self.update(ids=ids, masks=masks, frame_ids=frame_id)
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
