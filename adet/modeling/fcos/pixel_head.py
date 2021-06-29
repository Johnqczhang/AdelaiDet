from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec
from adet.modeling.fcos.fcos import Scale


def build_pixel_head(cfg, input_shape):
    return PixelHead(cfg, input_shape)


class PixelHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.PIXEL_HEAD.IN_FEATURES
        self.strides = cfg.MODEL.PIXEL_HEAD.FPN_STRIDES
        self.center_sample = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.radius = cfg.MODEL.FCOS.POS_RADIUS
        num_convs = cfg.MODEL.PIXEL_HEAD.NUM_CONVS
        in_channels = [input_shape[f].channels for f in self.in_features]
        in_channels = in_channels[0]

        tower = []
        for _ in range(num_convs):
            tower.append(nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, stride=1,
                padding=1, bias=True
            ))
            tower.append(nn.GroupNorm(32, in_channels))
            tower.append(nn.ReLU())
        self.add_module("mask_tower", nn.Sequential(*tower))

        # pixel embedding
        self.embed_reduce_factor = cfg.MODEL.PIXEL_HEAD.EMBED_REDUCE_FACTOR
        self.embed_dim = cfg.MODEL.PIXEL_HEAD.EMBED_DIM
        self.pixel_spatial_embed_pred = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.pixel_other_embed_pred = nn.Conv2d(
            in_channels, self.embed_dim - 2, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.position_scale = Scale(init_value=1.0)
        self.dist_func = cfg.MODEL.PIXEL_HEAD.EMBEDS_DIST_FUNC
        self.hinge_margins = cfg.MODEL.PIXEL_HEAD.HINGE_LOSS_MARGINS
        self.sample_ctr_on = cfg.MODEL.PIXEL_HEAD.SAMPLE_CTR_ON
        self.loss_intra_frame_on = cfg.MODEL.PIXEL_HEAD.LOSS_INTRA_FRAME_ON

        for m in [
            self.mask_tower,
            self.pixel_spatial_embed_pred, self.pixel_other_embed_pred
        ]:
            for l in m.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)

    def is_sample_mask_ctr(self):
        return self.sample_ctr_on == "mask"

    def forward(self, features, locations):
        self.pixel_embeds = []
        self.locations = locations
        for i in range(len(self.in_features)):
            x = self.mask_tower(features[i]) / self.embed_reduce_factor
            n, c, h, w = x.size()
            # coords: (n, 2, h, w)
            coords = locations[i].t().reshape(2, h, w).expand(n, -1, -1, -1)
            scaled_coords = self.position_scale(coords) / 100.0
            pixel_spatial_embed = self.pixel_spatial_embed_pred(x) + scaled_coords
            pixel_other_embed = self.pixel_other_embed_pred(x)
            pixel_embed = torch.cat([pixel_spatial_embed, pixel_other_embed], dim=1)
            pixel_embed = pixel_embed.permute(0, 2, 3, 1).reshape(n, -1, self.embed_dim)
            self.pixel_embeds.append(pixel_embed)

    def compute_embeds_distance(self, embed1, embed2):
        # embed1: (m, embed_dim)
        # embed2: (n, embed_dim)
        # return: dists, (m, n)
        if self.dist_func == "l2":
            dists = (embed1[:, None] - embed2[None]).square().sum(dim=-1).sqrt()
        elif self.dist_func == "cos":
            m1 = torch.linalg.norm(embed1, dim=1)
            m2 = torch.linalg.norm(embed2, dim=1)
            mod = (m1[:, None] * m2[None]).clip(min=1e-8)
            dists = 1. - embed1.matmul(embed2.t()) / mod
        else:
            raise ValueError(f"Incorrect embedding vector distance function: {self.dist_func}")
        return dists

    def losses(self, training_targets):
        pixel_embeds = self.pixel_embeds[0]
        n = pixel_embeds.shape[0]
        track_ids = training_targets["track_ids"]
        pxeb_losses = {}
        loss_pos, loss_neg = [], []
        loss_hard = []

        for i in range(0, n, 2):
            inds1 = (track_ids[i] != -1).nonzero().squeeze(1)
            inds2 = (track_ids[i + 1] != -1).nonzero().squeeze(1)
            if inds1.numel() == 0 or inds2.numel() == 0:
                continue

            dists = self.compute_embeds_distance(pixel_embeds[i, inds1], pixel_embeds[i + 1, inds2])
            t1, t2 = torch.meshgrid(track_ids[i][inds1], track_ids[i + 1][inds2])
            pos = t1 == t2
            # v1.0: hinge loss, count for all positive and negative pairs
            if self.hinge_margins[0] > 0 and pos.sum() > 0:
                loss_pos.append(F.relu(dists[pos] - self.hinge_margins[0]).square().mean())
            if self.hinge_margins[1] > 0 and pos.sum() < pos.numel():
                loss_neg.append(F.relu(self.hinge_margins[1] - dists[~pos]).square().mean())

            # v1.1: hard triplet loss, each location must have both positive and negative pairs
            if self.hinge_margins[2] > 0:
                dists_pos = dists * pos
                # since we want to find the minimum distance among negative pairs,
                # we set the distance of positive pairs to a very large value
                dists_neg = torch.where(pos, dists.new_tensor(10000.), dists)
                idx1 = (pos.sum(dim=1) > 0) & (pos.sum(dim=1) < dists.shape[0])
                idx2 = (pos.sum(dim=0) > 0) & (pos.sum(dim=0) < dists.shape[1])
                if idx1.sum() > 0:
                    loss = F.relu(dists_pos.amax(dim=1) - dists_neg.amin(dim=1) + self.hinge_margins[2]) * idx1
                    loss_hard.append(loss.sum() / idx1.sum())
                if idx2.sum() > 0:
                    loss = F.relu(dists_pos.amax(dim=0) - dists_neg.amin(dim=0) + self.hinge_margins[2]) * idx2
                    loss_hard.append(loss.sum() / idx2.sum())

        if self.hinge_margins[0] > 0:
            pxeb_losses["loss_pxeb_pos"] = sum(loss_pos) / len(loss_pos) if len(loss_pos) > 0 else pixel_embeds.sum() * 0.
        if self.hinge_margins[1] > 0:
            pxeb_losses["loss_pxeb_neg"] = sum(loss_neg) / len(loss_neg) if len(loss_neg) > 0 else pixel_embeds.sum() * 0.
        if self.hinge_margins[2] > 0:
            pxeb_losses["loss_pxeb_hard"] = sum(loss_hard) / len(loss_hard) if len(loss_hard) > 0 else pixel_embeds.sum() * 0.
        if self.loss_intra_frame_on:
            pxeb_losses.update(self.intra_frame_losses(pixel_embeds, track_ids))

        return pxeb_losses

    def intra_frame_losses(self, pixel_embeds, track_ids):
        losses = {}
        loss_pos, loss_neg, loss_hard = [], [], []

        for i, t_id in enumerate(track_ids):
            inds = (t_id != -1).nonzero().squeeze(1)
            if inds.numel() == 0:
                continue

            # symmetric matrix
            dists = self.compute_embeds_distance(pixel_embeds[i, inds], pixel_embeds[i, inds])
            t1, t2 = torch.meshgrid(t_id[inds], t_id[inds])
            pos = t1 == t2
            # v1.0: hinge loss, count for all positive and negative pairs
            if self.hinge_margins[0] > 0 and pos.sum() > 0:
                loss_pos.append(F.relu(dists[pos] - self.hinge_margins[0]).square().mean())
            if self.hinge_margins[1] > 0 and (~pos).sum() > 0:
                loss_neg.append(F.relu(self.hinge_margins[1] - dists[~pos]).square().mean())

            # v1.1: hard triplet loss
            if self.hinge_margins[2] > 0:
                loss = F.relu(dists[pos].amax() - dists[~pos].amin() + self.hinge_margins[2])
                loss_hard.append(loss)

        if self.hinge_margins[0] > 0:
            losses["loss_pxeb_intra_pos"] = sum(loss_pos) / len(loss_pos) if len(loss_pos) > 0 else pixel_embeds.sum() * 0.
        if self.hinge_margins[1] > 0:
            losses["loss_pxeb_intra_neg"] = sum(loss_neg) / len(loss_neg) if len(loss_neg) > 0 else pixel_embeds.sum() * 0.
        if self.hinge_margins[2] > 0:
            losses["loss_pxeb_intra_hard"] = sum(loss_hard) / len(loss_hard) if len(loss_hard) > 0 else pixel_embeds.sum() * 0.
        return losses

    def get_sample_region(self, boxes, strides, num_loc_list, loc_xs, loc_ys, bitmasks=None, radius=1):
        if bitmasks is not None:
            _, h, w = bitmasks.size()
            ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
            xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

            m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
            m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
            m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
            center_x = m10 / m00
            center_y = m01 / m00
        else:
            center_x = boxes[..., [0, 2]].sum(dim=-1) * 0.5
            center_y = boxes[..., [1, 3]].sum(dim=-1) * 0.5

        N = len(boxes)
        K = len(loc_xs)
        boxes = boxes[None].expand(K, N, 4)
        center_x = center_x[None].expand(K, N)
        center_y = center_y[None].expand(K, N)
        center_pt = boxes.new_zeros(boxes.shape)

        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in box
            center_pt[beg:end, :, 0] = torch.where(xmin > boxes[beg:end, :, 0], xmin, boxes[beg:end, :, 0])
            center_pt[beg:end, :, 1] = torch.where(ymin > boxes[beg:end, :, 1], ymin, boxes[beg:end, :, 1])
            center_pt[beg:end, :, 2] = torch.where(xmax > boxes[beg:end, :, 2], boxes[beg:end, :, 2], xmax)
            center_pt[beg:end, :, 3] = torch.where(ymax > boxes[beg:end, :, 3], boxes[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_pt[..., 0]
        right = center_pt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_pt[..., 1]
        bottom = center_pt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.amin(dim=-1) > 0
        return inside_gt_bbox_mask

    def postprocess(self, im_id, pred_instances):
        pred_instances = pred_instances[pred_instances.pred_classes == 0]
        if len(pred_instances) == 0:
            return pred_instances

        pixel_embeds = self.pixel_embeds[0]
        xs, ys = self.locations[0][:, 0], self.locations[0][:, 1]
        num_loc = len(xs)
        INF = 100000000
        pred_boxes = pred_instances.pred_boxes.tensor
        area = pred_instances.pred_boxes.area()
        l = xs[:, None] - pred_boxes[:, 0][None]
        t = ys[:, None] - pred_boxes[:, 1][None]
        r = pred_boxes[:, 2][None] - xs[:, None]
        b = pred_boxes[:, 3][None] - ys[:, None]
        reg_per_im = torch.stack([l, t, r, b], dim=2)

        if self.center_sample:
            bitmasks = pred_instances.bitmasks if pred_instances.has("bitmasks") else None
            is_in_boxes = self.get_sample_region(
                pred_boxes, self.strides, [num_loc], xs, ys,
                bitmasks=bitmasks, radius=self.radius
            )
        else:
            is_in_boxes = reg_per_im.amin(dim=2) > 0

        locations_to_area = area[None].repeat(num_loc, 1)
        locations_to_area[~is_in_boxes] = INF
        locations_to_min_area, locations_to_inds = locations_to_area.min(dim=1)
        locations_to_inds[locations_to_min_area == INF] = -1
        pred_instances.pixel_embeds = PixelEmbedsList([
            pixel_embeds[
                im_id,
                (locations_to_inds == i).nonzero()
            ].squeeze(1)
            for i in range(len(pred_instances))
        ])

        return pred_instances


class PixelEmbedsList:
    def __init__(self, embeds_list) -> None:
        self.embeds_list = embeds_list

    def __getitem__(self, item) -> "PixelEmbedsList":
        if item is None:
            return self
        if isinstance(item, int):
            if item >= len(self) or item < -len(self):
                raise IndexError("index out of range")
            else:
                return self.embeds_list[item]
        elif isinstance(item, torch.Tensor):
            if item.dtype == torch.bool:
                data = [self.embeds_list[i] for i, j in enumerate(item) if j]
            elif item.dtype == torch.int64:
                assert item.min() >= 0 and item.max() < len(self)
                data = [self.embeds_list[i] for i in item]
            return type(self)(data)
        else:
            raise IndexError("invalid index")

    def __len__(self) -> int:
        return len(self.embeds_list)

    def to(self, *args, **kwargs) -> "PixelEmbedsList":
        embeds_list = [
            e.to(*args, **kwargs) for e in self.embeds_list
        ]
        return type(self)(embeds_list)
