# -*- coding: utf-8 -*-
import logging
import math
import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict
from detectron2.layers import ShapeSpec
from adet.modeling.fcos.fcos import Scale
from fvcore.nn import sigmoid_focal_loss_jit

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
        self.use_margin = cfg.MODEL.EMBEDINST.USE_MARGIN
        if channels > 0:
            self.embed_conv = nn.Conv2d(
                in_channels, channels, kernel_size=1
            )
            convs = [self.embed_conv]
            if self.use_margin:
                self.margin_conv = nn.Conv2d(
                    in_channels, channels, kernel_size=1
                )
                convs.append(self.margin_conv)
            for m in convs:
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)
            in_channels = channels
        self.proposal_embedder = Embedder(in_channels, self.embed_dim)

        # learnable margin
        if self.use_margin:
            prior_margin = cfg.MODEL.EMBEDINST.PRIOR_MARGIN
            init_margin_bias = math.log(math.log(2) / (prior_margin ** 2))
            self.margin_embedder = Embedder(in_channels, self.embed_dim)
            for l in self.margin_embedder.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.constant_(l.bias, init_margin_bias)
            self.margin_scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(input_shape))])
            self.margin_reduce_factor = cfg.MODEL.EMBEDINST.MARGIN_REDUCE_FACTOR

        # re-ID branch
        self.loss_reid_on = cfg.MODEL.EMBEDINST.LOSS_REID_ON
        if self.loss_reid_on:
            num_iids = cfg.MODEL.EMBEDINST.NUM_INST_IDS
            in_channels = self.embed_dim * 2 if self.use_margin else self.embed_dim
            # cls_logits = nn.Conv2d(
            #     in_channels, num_iids, kernel_size=1
            # )
            cls_logits = nn.Linear(in_channels, num_iids)
            nn.init.normal_(cls_logits.weight, std=0.01)
            bias_value = 0
            self.loss_reid_type = cfg.MODEL.EMBEDINST.LOSS_REID_TYPE
            if self.loss_reid_type == "focal":
                # initialize the bias for focal loss
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(cls_logits.bias, bias_value)
            self.cls_logits = cls_logits
            self.reid_scale = math.sqrt(2) * math.log(num_iids)

        # hyper-params
        self.loss_w = {
            "hinge": cfg.MODEL.EMBEDINST.LOSS_WEIGHT_HINGE
        }
        self.loss_smooth_on = cfg.MODEL.EMBEDINST.LOSS_SMOOTH_ON
        if self.loss_smooth_on:
            self.loss_w["smooth"] = cfg.MODEL.EMBEDINST.LOSS_WEIGHT_SMOOTH
        if self.loss_reid_on:
            self.loss_w["reid"] = cfg.MODEL.EMBEDINST.LOSS_WEIGHT_REID
        # self.loss_intra_seq_on = cfg.MODEL.EMBEDINST.LOSS_INTRA_SEQ_ON

    def forward(self, proposals):
        tower_feats = proposals["tower_feats"]
        locations = proposals["locations"]
        pred_embeds = []
        # inst_logits = []
        for i, f in enumerate(self.in_features):
            embed_x = tower_feats[f]
            if hasattr(self, "embed_conv"):
                embed_x = self.embed_conv(embed_x)
            embeds = self.proposal_embedder(embed_x, locations=locations[i])

            if self.use_margin:
                margin_x = tower_feats[f] / self.margin_reduce_factor
                if hasattr(self, "margin_conv"):
                    margin_x = self.margin_conv(margin_x)
                margins = self.margin_embedder(margin_x, scales=self.margin_scales[i])
                # Here, the network predicts 1/(2 x sigma^2) directly
                margins = margins.exp()
                embeds = torch.cat([embeds, margins], dim=1)

            pred_embeds.append(embeds)
            # if self.loss_reid_on:
            #     inst_logits.append(self.cls_logits(embeds))

        pred_embeds = torch.cat([
            # reshape: (N, embed_dim, Hi, Wi) -> (N*Hi*Wi, embed_dim), level first
            x.permute(0, 2, 3, 1).reshape(-1, x.size(1)) for x in pred_embeds
        ], dim=0)
        pos_inds = proposals["instances"].pos_inds
        proposals["instances"].pred_embeds = pred_embeds[pos_inds]

        if self.loss_reid_on:
            # proposals["instances"].inst_logits = torch.cat([
            #     x.permute(0, 2, 3, 1).reshape(-1, x.size(1)) for x in inst_logits
            # ], dim=0)[pos_inds]
            reid_feats = self.reid_scale * F.normalize(pred_embeds[pos_inds])
            proposals["instances"].inst_logits = self.cls_logits(reid_feats)

        return proposals

    def sample_proposals(self, pred_instances):
        # sample positive proposals in which pred_box has IoU > 0.5 with the matched gt instance
        # TODO:
        return pred_instances

    def losses(self, pred_instances):
        loss = {}
        if len(pred_instances) == 0:
            dummy_loss = pred_instances.pred_embeds.sum() * 0
            loss["loss_prop_hinge"] = dummy_loss
            if self.loss_smooth_on:
                loss["loss_prop_smooth"] = dummy_loss
            if self.loss_reid_on:
                loss["loss_prop_reid"] = dummy_loss
            return loss

        loss["loss_prop_hinge"] = self.hinge_loss(pred_instances) * self.loss_w["hinge"]
        if self.loss_smooth_on:
            loss["loss_prop_smooth"] = self.smooth_loss(pred_instances) * self.loss_w["smooth"]
        if self.loss_reid_on:
            loss["loss_prop_reid"] = self.reid_loss(pred_instances) * self.loss_w["reid"]

        return loss

    def compute_dist_probs(self, emb1, emb2):
        if self.use_margin:
            emb1, emb2 = emb1[:, :self.embed_dim], emb2[:, :self.embed_dim]
            var1, var2 = emb1[:, self.embed_dim:], emb2[:, self.embed_dim:]
            vars = 4 * var1 * var2 / (var1 + var2)
        else:
            vars = 0.5

        dists = F.mse_loss(emb1, emb2, reduction="none")        
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
            pred_embeds[inds[:, 0]], pred_embeds[inds[:, 1]]
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
        return loss

    def reid_loss(self, pred_instances):
        # convert to 0-based id
        inst_ids = pred_instances.inst_ids - 1
        if self.loss_reid_type == "focal":
            target = torch.zeros_like(pred_instances.inst_logits)
            target = target.scatter(1, inst_ids[:, None], 1)
            id_loss = sigmoid_focal_loss_jit(
                pred_instances.inst_logits, target,
                alpha=0.25, gamma=2.0, reduction="sum"
            ) / inst_ids.size(0)
        else:
            id_loss = F.cross_entropy(
                pred_instances.inst_logits, inst_ids,
            )
        return id_loss


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
