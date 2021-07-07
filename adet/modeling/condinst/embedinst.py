# -*- coding: utf-8 -*-
import logging
import math
import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict
from detectron2.layers import ShapeSpec
from adet.modeling.fcos.fcos import Scale

from .mask_branch import build_mask_branch


logger = logging.getLogger(__name__)


def build_embedinst(cfg, input_shape):
    if cfg.MODEL.EMBEDINST.ENABLED:
        return EmbedInst(cfg, input_shape)
    return None


class EmbedInst(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # proposal embedding
        self.proposal_embedder = build_proposal_embedder(cfg, input_shape)
        # pixel embedding
        self.pixel_embedder = build_pixel_embedder(cfg, input_shape)

    def forward(self, proposals, features, mask_feats):
        assert "locations" in proposals

        if self.proposal_embedder:
            proposals = self.proposal_embedder(proposals)

        if self.pixel_embedder:
            # only predict pixel embeddings on P3-level feature map
            self.pixel_embedder(
                proposals["locations"][0], features, mask_feats
            )

        return proposals

    def losses(self, pred_instances, gt_instances):
        loss_embeds = {}
        if self.proposal_embedder:
            loss_embeds.update(
                self.proposal_embedder.losses(pred_instances)
            )

        if self.pixel_embedder:
            loss_embeds.update(
                self.pixel_embedder.losses(pred_instances, gt_instances)
            )

        return loss_embeds


def build_proposal_embedder(cfg, input_shape):
    if cfg.MODEL.EMBEDINST.PROPOSAL_HEAD_CHANNELS < 0:
        return None
    return ProposalEmbedder(cfg, input_shape)


def lovasz_grad(y):
    """
    Computes gradient of the Lovasz extension w.r.t. sorted errors
    y (torch.LongTensor): (N, D): binary labels sorted according to descending errors.
    """
    n = y.sum(dim=1, keepdim=True)
    intersection = n - y.cumsum(dim=1)
    union = n + (1 - y).cumsum(dim=1)
    jaccard = 1 - intersection / union
    p = y.size(1)
    if p > 1:  # cover 1-pixel case
        jaccard[:, 1:p] = jaccard[:, 1:p] - jaccard[:, 0:-1]
    return jaccard


def lovasz_hinge(x, y):
    # x, y: (N, D)
    signs = (2 * y - 1).float()
    err = 1 - x * signs
    # `stable=True` only works on the CPU when torch>=1.9.0 for now
    # _, inds = err.cpu().sort(dim=1, descending=True, stable=True)
    # err_desc = err.gather(1, inds)
    err_desc, inds = err.sort(dim=1, descending=True)
    y_desc = y.gather(1, inds)
    grad = lovasz_grad(y_desc.float())
    loss = (F.relu(err_desc) * grad).sum(dim=1).mean()
    return loss


def lovasz_loss(x, y):
    eps = 1e-6
    x = x.clip(min=eps, max=1-eps)
    x = x.log() - (1 - x).log()
    return lovasz_hinge(x, y)


class ProposalEmbedder(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.embed_dim = cfg.MODEL.EMBEDINST.EMBED_DIM
        in_channels = [v.channels for v in input_shape.values()][0]
        channels = cfg.MODEL.EMBEDINST.PROPOSAL_HEAD_CHANNELS
        if channels > 0:
            self.embed_conv = nn.Conv2d(
                in_channels, channels, kernel_size=1
            )
            self.margin_conv = nn.Conv2d(
                in_channels, channels, kernel_size=1
            )
            for m in [self.embed_conv, self.margin_conv]:
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
            in_channels = channels
        self.proposal_embedder = Embedder(in_channels, self.embed_dim)

        # learnable margin
        prior_margin = cfg.MODEL.EMBEDINST.PRIOR_MARGIN
        init_margin_bias = math.log(math.log(2) / (prior_margin ** 2))
        self.margin_embedder = Embedder(in_channels, self.embed_dim)
        for l in self.margin_embedder.modules():
            if isinstance(l, nn.Conv2d):
                nn.init.constant_(l.bias, init_margin_bias)
        self.margin_scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(input_shape))])
        self.margin_reduce_factor = cfg.MODEL.EMBEDINST.MARGIN_REDUCE_FACTOR

        # hyper-params
        self.loss_smooth_on = cfg.MODEL.EMBEDINST.LOSS_SMOOTH_ON
        self.loss_w_smooth = cfg.MODEL.EMBEDINST.LOSS_WEIGHT_SMOOTH
        # self.loss_intra_seq_on = cfg.MODEL.EMBEDINST.LOSS_INTRA_SEQ_ON

    def forward(self, proposals):
        box_feats = proposals["box_feats"]
        locations = proposals["locations"]
        pred_embeds = []
        for i, f in enumerate(self.in_features):
            embed_x = box_feats[f]
            margin_x = box_feats[f] / self.margin_reduce_factor
            if hasattr(self, "embed_conv"):
                embed_x = self.embed_conv(embed_x)
                margin_x = self.margin_conv(margin_x)

            margins = self.margin_embedder(margin_x, scales=self.margin_scales[i])
            # Here, the network predicts 1/(2 x sigma^2) directly
            margins = margins.exp()
            embeds = self.proposal_embedder(embed_x, locations=locations[i])
            pred_embeds.append(torch.cat([embeds, margins], dim=1))

        pred_embeds = torch.cat([
            # reshape: (N, embed_dim, Hi, Wi) -> (N*Hi*Wi, embed_dim), level first
            x.permute(0, 2, 3, 1).reshape(-1, x.size(1)) for x in pred_embeds
        ], dim=0)
        pos_inds = proposals["instances"].pos_inds
        proposals["instances"].pred_embeds = pred_embeds[pos_inds]

        return proposals

    def sample_proposals(self, pred_instances):
        # sample positive proposals in which pred_box has IoU > 0.5 with the matched gt instance
        # TODO:
        return pred_instances

    def losses(self, pred_instances):
        loss = {}
        if len(pred_instances) == 0:
            dummy_loss = self.proposal_feats.sum() * 0
            loss["loss_prop_hinge"] = dummy_loss
            if self.loss_smooth_on:
                loss["loss_prop_smooth"] = dummy_loss
            return loss

        loss["loss_prop_hinge"] = self.hinge_loss(pred_instances)
        if self.loss_smooth_on:
            loss["loss_prop_smooth"] = self.smooth_loss(pred_instances)

        return loss

    def compute_dist_probs(self, p1, p2, m1, m2):
        dists = F.mse_loss(p1, p2, reduction="none")
        vars = 4 * m1 * m2 / (m1 + m2)
        probs = (-(dists * vars).sum(dim=-1)).exp()
        return probs

    def hinge_loss(self, pred_instances):
        inst_ids = pred_instances.inst_ids
        pred_embeds = pred_instances.pred_embeds
        num_insts = len(inst_ids)
        # construct pairwise indices, shape: (N, 2)
        inds = torch.combinations(
            torch.arange(num_insts, device=inst_ids.device), r=2
        )
        # compute distance and map it to the probability, shape: (N,)
        probs = self.compute_dist_probs(
            pred_embeds[inds[:, 0], :self.embed_dim],
            pred_embeds[inds[:, 1], :self.embed_dim],
            pred_embeds[inds[:, 0], self.embed_dim:],
            pred_embeds[inds[:, 1], self.embed_dim:]
        )
        # find indices of positive and negative samples
        pos = inst_ids[inds[:, 0]] == inst_ids[inds[:, 1]]
        # shape: (num_insts, num_insts - 1)
        inds_per_inst = [
            (inds == i).nonzero(as_tuple=True)[0] for i in range(num_insts)
        ]
        # the i-th row indicates whether the i-th proposal has the same inst_id with all other proposals
        pos_per_inst = torch.stack([
            pos[per_inds] for per_inds in inds_per_inst
        ], dim=0)
        probs_per_inst = torch.stack([
            probs[per_inds] for per_inds in inds_per_inst
        ], dim=0)

        return lovasz_loss(probs_per_inst, pos_per_inst)

    def smooth_loss(self, pred_instances):
        pred_embeds = pred_instances.pred_embeds
        inst_ids = pred_instances.inst_ids
        uniq_iids = pred_instances.inst_ids.unique()
        # construct one-hot matrix (num_proposals, num_inst_ids)
        one_hot = inst_ids[:, None] == uniq_iids[None]
        num_iids = one_hot.sum(dim=0)
        # shape: (num_proposals, num_inst_ids, embed_dim)
        mean_embeds = pred_embeds[:, None] * one_hot[..., None]
        # shape: (num_inst_ids, embed_dim)
        mean_embeds = mean_embeds.sum(dim=0) / num_iids[:, None]
        loss = (pred_embeds[:, None] - mean_embeds[None]).square().sum(dim=-1) * one_hot
        loss = (loss.sum(dim=0) / num_iids).mean()
        return loss * self.loss_w_smooth


def compute_distances(func, a, b, reduction="none"):
    if func == "l2":
        return distance_l2(a, b, reduction)
    elif func == "cos":
        return distance_cos(a, b, reduction)


def distance_l2(a, b, reduction="none"):
    dists = (a - b).square()
    if reduction == "sum":
        dists = dists.sum(dim=-1)
    elif reduction == "sqrt":
        dists = dists.sum(dim=-1).sqrt()

    return dists


def distance_cos(a, b):
    m1 = torch.linalg.norm(a, dim=1)
    m2 = torch.linalg.norm(b, dim=1)
    mod = (m1[:, None] * m2[None]).clip(min=1e-8)
    return 1 - a.matmul(b.t()) / mod


def build_pixel_embedder(cfg, input_shape):
    if not cfg.MODEL.EMBEDINST.PIXEL_BRANCH.ENABLED:
        return None
    return PixelEmbedder(cfg, input_shape)


class PixelEmbedder(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.embed_dim = cfg.MODEL.EMBEDINST.EMBED_DIM
        if cfg.MODEL.EMBEDINST.PIXEL_BRANCH.SHARED:
            self.pixel_branch = None
            in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        else:
            _cfg = cfg.clone()
            _cfg.defrost()
            _cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS = cfg.MODEL.EMBEDINST.PIXEL_BRANCH.OUT_CHANNELS
            _cfg.freeze()
            self.pixel_branch = build_mask_branch(_cfg, input_shape)
            in_channels = cfg.MODEL.EMBEDINST.PIXEL_BRANCH.OUT_CHANNELS

        self.embedder = Embedder(in_channels, self.embed_dim)

    def forward(self, locations, features, mask_feats):
        self.locations = locations  # TODO:
        if self.pixel_branch:
            pixel_feats, _ = self.pixel_branch(features)
        else:
            pixel_feats = mask_feats

        pred_embeds = self.embedder(pixel_feats, locations=locations)
        # reshape: (N, embed_dim, H, W) -> (N*H*W, embed_dim)
        self.pred_embeds = pred_embeds.permute(0, 2, 3, 1).reshape(pred_embeds.size(0), -1, self.embed_dim)

    def losses(self, pred_instances, gt_instances):
        loss_embeds = {}

        return loss_embeds


class Embedder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.spatial_embed_pred = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1, padding=1
        )
        nn.init.normal_(self.spatial_embed_pred.weight, std=0.01)
        nn.init.constant_(self.spatial_embed_pred.bias, 0)

        self.free_embed_pred = nn.Conv2d(
            in_channels, out_channels - 2, kernel_size=3, stride=1, padding=1
        )
        nn.init.normal_(self.free_embed_pred.weight, std=0.01)
        nn.init.constant_(self.free_embed_pred.bias, 0)

    def forward(self, feats, locations=None, scales=None):
        spatial_embeds = self.spatial_embed_pred(feats)
        assert (locations is None) or (scales is None), \
            "locations and scales cannot be used at the same time"
        if scales is not None:
            spatial_embeds = scales(spatial_embeds)
        elif locations is not None:
            h, w = feats.shape[-2:]
            scaled_coords = locations.t().reshape(2, h, w) / 100.0
            # scaled_coords = self.position_scale(coords) / 100.0
            spatial_embeds = spatial_embeds + scaled_coords
        else:
            raise NotImplementedError

        free_embeds = self.free_embed_pred(feats)
        return torch.cat([spatial_embeds, free_embeds], dim=1)
