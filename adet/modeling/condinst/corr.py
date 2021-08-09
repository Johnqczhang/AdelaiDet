import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def build_corr_block(cfg):
    if cfg.MODEL.PX_VOLUME.ENABLED:
        return CorrBlock(cfg)
    return None


def mask_feats_pooler(fmap, masks, pooler_type="mean"):
    """
    Extract a feature vector for each instance by pooling on mask feature map
    over all foreground pixels indicated by masks.

    Args:
        fmap (C, H, W): a (normalized and reshaped) mask feature map for an image output by the mask branch
        masks (N, H, W): the ground-truth or predicted masks of N instances 

    Returns:
        feats (N, C): pooled C-dim feature vectors for N instances
    """
    feats = torch.stack([mask[None] * fmap for mask in masks], dim=0)  # (N, C, H, W)
    if pooler_type == "avg":
        return feats.sum(dim=(2, 3)) / masks.sum(dim=(1, 2)).clip(min=1)[:, None]
    elif pooler_type == "max":
        return feats.amax(dim=(2, 3))
    else:
        raise NotImplementedError


class CorrBlock:
    def __init__(self, cfg):
        self.vec_op_on = cfg.MODEL.PX_VOLUME.VEC_OP_ON
        self.norm_feats_on = cfg.MODEL.PX_VOLUME.NORM_FEATS_ON
        self.T = cfg.MODEL.PX_VOLUME.LOSS_NCE_TEMP
        self.loss_w = cfg.MODEL.PX_VOLUME.LOSS_WEIGHT
        self.pooler_type = cfg.MODEL.PX_VOLUME.MASK_POOLER_TYPE

    def build_pixel_corr(self, mask_feats):
        """
        Construct 4D correlation volume for all pairs of pixels

        Args:
            mask_feats (2n, h*w, c)
        Returns:
            corr (n, h*w, h*w)
        """
        n = mask_feats.size(0) // 2
        # Einstein sum is more intuitive (i.e., batch matrix multiplication)
        corr = torch.einsum("npc,nqc->npq", [mask_feats[:n], mask_feats[n:]])
        return corr

    def loss(self, mask_feats, mask_feat_stride, gt_instances):
        # mask_feats: (N, C, H, W)
        # normalization over the feature dimension
        if self.norm_feats_on:
            mask_feats = F.normalize(mask_feats, dim=1)

        n = mask_feats.size(0) // 2
        loss_corr = []
        for i in range(n):
            j = i + n
            if len(gt_instances[i]) == 0 or len(gt_instances[j]) == 0:
                continue

            if self.pooler_type == "ctr":
                ctr1 = (gt_instances[i].gt_centers / mask_feat_stride).long()
                ctr2 = (gt_instances[j].gt_centers / mask_feat_stride).long()
                feats1 = mask_feats[i, :, ctr1[:, 1], ctr1[:, 0]].t()
                feats2 = mask_feats[j, :, ctr2[:, 1], ctr2[:, 0]].t()
            else:
                gt_masks1 = gt_instances[i].gt_bitmasks_p3  # (N1, H, W)
                gt_masks2 = gt_instances[j].gt_bitmasks_p3  # (N2, H, W)
                feats1 = mask_feats_pooler(mask_feats[i], gt_masks1, self.pooler_type)
                feats2 = mask_feats_pooler(mask_feats[j], gt_masks2, self.pooler_type)

            corr = torch.einsum("pc,qc->pq", [feats1, feats2]) / self.T

            gt_ids1 = gt_instances[i].gt_corr_ids[:, 0]
            gt_ids2 = gt_instances[j].gt_corr_ids[:, 0]
            loss1 = F.cross_entropy(corr, gt_ids1, ignore_index=-1)
            loss2 = F.cross_entropy(corr.t(), gt_ids2, ignore_index=-1)
            loss_corr.extend([loss1, loss2])

        if len(loss_corr) == 0:
            loss_corr = mask_feats.sum() * 0
        else:
            loss_corr = sum(loss_corr) / len(loss_corr) * self.loss_w

        return dict(loss_corr=loss_corr)

    def postprocess(self, mask_fmap, mask_feat_stride, pred_instances):
        # mask_fmap: (C, H, W)
        if self.norm_feats_on:
            mask_fmap = F.normalize(mask_fmap, dim=0)
        if self.pooler_type == "ctr":
            ctrs = get_mask_centers(pred_instances.pred_global_masks)
            ctrs = (ctrs / mask_feat_stride).long()
            mfeats = mask_fmap[:, ctrs[:, 1], ctrs[:, 0]].t()
        else:
            pred_masks = pred_instances.pred_p3_masks  # (N, H, W)
            mfeats = mask_feats_pooler(mask_fmap, pred_masks, self.pooler_type)

        pred_instances.pred_mfeats = mfeats
        return pred_instances


def get_mask_centers(masks):
    h, w = masks.size()[1:3]
    ys = torch.arange(0, h, dtype=torch.float32, device=masks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=masks.device)

    m00 = masks.sum(dim=(1, 2)).clip(min=1e-6)
    m10 = (masks * xs).sum(dim=(1, 2))
    m01 = (masks * ys[:, None]).sum(dim=(1, 2))
    centers = torch.stack([m10, m01], dim=1) / m00[:, None]

    return centers
