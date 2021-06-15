from typing import Dict

import torch
from torch import nn

from detectron2.layers import ShapeSpec
from adet.modeling.fcos.fcos import Scale


def build_pixel_head(cfg, input_shape):
    return PixelHead(cfg, input_shape)


class PixelHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.PIXEL_HEAD.IN_FEATURES
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

        for m in [
            self.mask_tower,
            self.pixel_spatial_embed_pred, self.pixel_other_embed_pred
        ]:
            for l in m.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)

    def forward(self, features, locations):
        pixel_embeds = []
        for i in range(len(self.in_features)):
            x = self.mask_tower(features[i]) / self.embed_reduce_factor
            n, c, h, w = x.size()
            # coords: (n, 2, h, w)
            coords = locations[i].t().reshape(2, h, w).expand(n, -1, -1, -1)
            scaled_coords = self.position_scale(coords) / 100.0
            pixel_spatial_embed = self.pixel_spatial_embed_pred(x) + scaled_coords
            pixel_other_embed = self.pixel_other_embed_pred(x)
            pixel_embed = torch.cat([pixel_spatial_embed, pixel_other_embed], dim=1)
            pixel_embeds.append(pixel_embed)

        return pixel_embeds

    def compute_embeds_distance(self, embed1, embed2):
        # embed1: (m, embed_dim)
        # embed2: (n, embed_dim)
        # return: dists, (m, n)
        if self.dist_func == "ed":
            dists = (embed1[:, None] - embed2[None]).square().sum(dim=-1).sqrt()
        elif self.dist_func == "cos":
            m1 = torch.linalg.norm(embed1, dim=1)
            m2 = torch.linalg.norm(embed2, dim=1)
            mod = (m1[:, None] * m2[None]).clip(min=1e-8)
            dists = 1. - embed1.matmul(embed2.t()) / mod
        else:
            raise ValueError(f"Incorrect embedding vector distance function: {self.dist_func}")
        return dists

    def compute_losses(self, pixel_embeds, training_targets):
        losses = {}
        n, c = pixel_embeds[0].shape[:2]
        pixel_embeds = pixel_embeds[0].permute(0, 2, 3, 1).reshape(n, -1, c)
        track_ids = training_targets["track_ids"]
        loss_pos, loss_neg = [], []

        for i in range(0, n, 2):
            inds1 = (track_ids[i] != -1).nonzero().squeeze(1)
            inds2 = (track_ids[i + 1] != -1).nonzero().squeeze(1)
            if inds1.numel() == 0 or inds2.numel() == 0:
                continue

            dists = self.compute_embeds_distance(pixel_embeds[i, inds1], pixel_embeds[i + 1, inds2])
            t1, t2 = torch.meshgrid(track_ids[i][inds1], track_ids[i + 1][inds2])
            pos = t1 == t2
            # hinge loss v1.0: count for all positive and negative pairs
            if pos.sum() > 0:
                loss_pos.append((dists[pos] - self.hinge_margins[0]).clip(min=0).square().mean())
            if pos.sum() < pos.numel():
                loss_neg.append((self.hinge_margins[1] - dists[~pos]).clip(min=0).square().mean())

        losses["loss_pixel_embed_pos"] = sum(loss_pos) / len(loss_pos) if len(loss_pos) > 0 else pixel_embeds.sum() * 0.
        losses["loss_pixel_embed_neg"] = sum(loss_neg) / len(loss_neg) if len(loss_neg) > 0 else pixel_embeds.sum() * 0.

        return losses

    def assign_pixels_to_proposals(self, proposals, locations, pixel_embeds_pred):
        xs, ys = locations[:, 0], locations[:, 1]
        num_loc = len(xs)
        INF = 100000000
        n, c = pixel_embeds_pred.shape[:2]
        pixel_embeds_pred = pixel_embeds_pred.permute(0, 2, 3, 1).reshape(n, -1, c)

        for im_id, per_im in enumerate(proposals):
            pred_boxes = per_im.pred_boxes.tensor
            if pred_boxes.numel() == 0:
                continue

            area = per_im.pred_boxes.area()
            l = xs[:, None] - pred_boxes[:, 0][None]
            t = ys[:, None] - pred_boxes[:, 1][None]
            r = pred_boxes[:, 2][None] - xs[:, None]
            b = pred_boxes[:, 3][None] - ys[:, None]
            reg_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_per_im.min(dim=2)[0] > 0
            locations_to_area = area[None].repeat(num_loc, 1)
            locations_to_area[~is_in_boxes] = INF
            locations_to_min_area, locations_to_inds = locations_to_area.min(dim=1)

            pred_classes = per_im.pred_classes[locations_to_inds]
            pred_classes[locations_to_min_area == INF] = 80
            locations_to_inds[pred_classes != 0] = -1
            per_im.pixel_embeds = PixelEmbedsList([
                pixel_embeds_pred[
                    im_id,
                    (locations_to_inds == i).nonzero()
                ].squeeze(1)
                for i in range(len(per_im))
            ])

        return proposals


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
