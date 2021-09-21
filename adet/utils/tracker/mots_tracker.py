import torch
import torch.nn.functional as F

from motmetrics.lap import linear_sum_assignment

from detectron2.structures.instances import Instances
from adet.structures.tracks import Tracks, TrackState
from .motion import KalmanFilter


def xyxy2xyah(boxes):
    """ Convert bounding boxes from xyxy to xyah.

    Args:
        boxes (Tensor[float]): a Nx4 matrix. Each row is (x1, y1, x2, y2).

    Returns:
        boxes (Tensor[float]): a Nx4 matrix. Each row is (x, y, a, h) with
            center position (x, y), aspect ratio a, and height h.
    """
    center = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    ah = boxes[:, 2:4] - boxes[:, 0:2]
    ah[:, 0] /= ah[:, 1]
    xyah = torch.cat([center, ah], dim=1)
    return xyah


def pairwise_miou(masks1: torch.Tensor, masks2: torch.Tensor) -> torch.Tensor:
    """
    Given two lists of binary masks of size M and N, compute the IoU between all M x N pairs of masks.

    Args:
        masks1, masks2 (Tensor): two tensor containing M & N binary masks, respectively.

    Returns:
        ious (Tensor): (M, N)
    """
    inter = (masks1[:, None] * masks2).sum(dim=(2, 3))
    union = (masks1[:, None] + masks2).clamp(min=0, max=1).sum(dim=(2, 3))

    # handle empty masks
    ious = torch.where(
        union > 0,
        inter / union,
        inter.new_zeros(1)
    )
    return ious


def pairwise_biou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Given two lists of boxes of size M and N, compute the IoU between all M x N pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1, boxes2 (Tensor): two tensor containing M & N boxes, respectively.

    Returns:
        ious (Tensor): (M, N)
    """
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).prod(dim=1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).prod(dim=1)
    inter = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter = inter.clamp(min=0).prod(dim=2)

    ious = inter / (area1[:, None] + area2 - inter)
    return ious


class MOTSTracker(object):
    """ A simple tracker for multi-object tracking and segmentation.

    Args:
        cfg: configurations
        frame_rate (int): number of frames per second.
    """
    def __init__(self, cfg, frame_rate=30) -> None:
        self.cfg = cfg
        self.num_frames_confirm = cfg.num_tentative
        self.memo_items = None
        if cfg.with_motion:
            self.kf = KalmanFilter()

        self.reset(frame_rate)

    def reset(self, frame_rate=30) -> None:
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.num_frames_retain = frame_rate
        self.tracks = Tracks(ids=torch.arange(self.num_tracks))

    @property
    def empty(self):
        """Whether the set of tracks is empty or not."""
        return len(self.tracks) == 0

    @property
    def ids(self):
        """All ids in the tracker."""
        return self.tracks.ids

    def instantiate(self, instances) -> "Tracks":
        """Instantiate predicted instances as new tracks."""
        items = {}
        # items["scores"] = instances.get("scores")
        # items["classes"] = instances.get("pred_classes")
        # assert instances.has("pred_masks")
        if self.cfg.with_miou:
            items["masks"] = instances.get("pred_masks")
        if self.cfg.with_biou:
            items["boxes"] = instances.get("pred_boxes").tensor
        if self.cfg.with_motion:
            items["xyah"] = xyxy2xyah(items["boxes"])
        if self.cfg.with_embeds:
            assert instances.has("pred_embeds")
            items["embeds"] = F.normalize(instances.get("pred_embeds"), dim=1)

        assert len(items) > 0, f"Error: Got empty measurements with args: {self.cfg}"

        if self.memo_items is None:
            self.memo_items = list(items.keys())
        else:
            assert self.memo_items == list(items.keys())

        # initialize track ids and indices of matched tracks to -1
        num_insts = len(instances)
        items["ids"] = instances.im_inds.new_full((num_insts,), -1)
        items["match_inds"] = instances.im_inds.new_full((num_insts,), -1)

        return Tracks(**items)

    def init_tracks(self, frame_id: int, new_tracks: "Tracks") -> None:
        """Create new tracks (tracklets)."""
        num_tracks = len(new_tracks)
        new_tracks.state = new_tracks.ids.new_full((num_tracks,), TrackState.Tentative)
        new_tracks.start_frame = new_tracks.ids.new_full((num_tracks,), frame_id)
        new_tracks.end_frame = new_tracks.ids.new_full((num_tracks,), frame_id)
        new_tracks.remove("match_inds")

        if self.cfg.with_motion:
            new_tracks.mean, new_tracks.cov = self.kf.initiate(new_tracks.xyah)

        if self.cfg.with_embeds:
            new_tracks.embeds = F.normalize(new_tracks.embeds, dim=1)

        if self.empty:
            self.tracks = new_tracks
            # self.tracks.state[:] = TrackState.Tracked
        else:
            self.tracks = Tracks.cat([self.tracks, new_tracks])

        self.num_tracks += num_tracks

    def update_embeds(self, embeds: torch.Tensor, cur_embeds: torch.Tensor) -> "torch.Tensor":
        """
        """
        cur_embeds = F.normalize(cur_embeds, dim=1)
        if self.cfg.momentums is not None and "embeds" in self.cfg.momentums:
            m = self.cfg.momentums["embeds"]
            embeds = m * embeds + (1 - m) * cur_embeds

        return embeds

    def update_tracks(self, frame_id: int, instances: "Tracks") -> None:
        """ Update tracks that have been matched in the current frame."""
        match_inds = instances.match_inds
        # instances.remove("match_inds")

        if self.cfg.with_motion:
            mean, cov = self.tracks.mean, self.tracks.cov
            mean[match_inds], cov[match_inds] = self.kf.update(
                mean[match_inds], cov[match_inds], instances.xyah
            )
        if self.cfg.with_embeds:
            embeds = self.tracks.embeds
            embeds[match_inds] = self.update_embeds(embeds[match_inds], instances.embeds)

        # Update measurements
        for item in self.memo_items:
            if item in ["ids", "match_inds"]:
                continue
            self.tracks.get(item)[match_inds] = instances.get(item)

        self.tracks.end_frame[match_inds] = frame_id

    def update_state(self, frame_id: int) -> None:
        """ Update state of tracks in the tracker."""
        matched = self.tracks.end_frame == frame_id
        lost = self.tracks.state == TrackState.Lost
        tracked = self.tracks.state == TrackState.Tracked
        tentative = self.tracks.state == TrackState.Tentative

        """ Mark lost tracks re-matched in the current frame as Tracked."""
        self.tracks.state[lost & matched] = TrackState.Tracked

        """ Mark tracked tracks unmatched in the current frame as Lost."""
        self.tracks.state[tracked & (~matched)] = TrackState.Lost

        """ Mark tentative tracks matched consecutively over `num_frames_confirm` frames as Tracked."""
        confirmed = (frame_id - self.tracks.start_frame) >= self.num_frames_confirm
        self.tracks.state[tentative & matched & confirmed] = TrackState.Tracked
        """ Mark tentative tracks unmatched in the current frame as Removed."""
        self.tracks.state[tentative & (~matched)] = TrackState.Removed

        """ Mark tracks unmatched consecutively over `num_frames_retain` frames as Removed."""
        removed = (frame_id - self.tracks.end_frame) >= self.num_frames_retain
        self.tracks.state[removed] = TrackState.Removed

    def remove_duplicate_tracks(self) -> None:
        """ Remove duplicate tracks that have large bounding box overlap between each other."""
        state = self.tracks.state
        active_inds = ((state == TrackState.Tracked) | (state == TrackState.Tentative)).nonzero().squeeze(1)
        lost_inds = (state == TrackState.Lost).nonzero().squeeze(1)
        if active_inds.numel() == 0 or lost_inds.numel() == 0:
            return

        tracks_length = self.tracks.end_frame - self.tracks.start_frame
        ious = pairwise_biou(self.tracks.boxes[active_inds], self.tracks.boxes[lost_inds])
        row, col = (ious > 0.85).nonzero().t()
        dup_inds = torch.where(
            tracks_length[row] >= tracks_length[col],
            lost_inds[col],
            active_inds[row]
        )
        if dup_inds.numel() > 0:
            print(f"Remove {dup_inds.numel()} duplicate tracks.")
        self.tracks.state[dup_inds] = TrackState.Removed

    def remove_invalid_tracks(self) -> None:
        """ Remove invalid tracks (state = Removed)."""
        if self.cfg.with_deduplication:
            self.remove_duplicate_tracks()
        self.tracks = self.tracks[self.tracks.state != TrackState.Removed]

    def update(self, frame_id: int, instances: "Tracks") -> None:
        """ Update the tracker.

        Args:
            frame_id (int): The id of current frame, 0-index.
            instances (Tracks): The tracklets of instances in current frame.
        """
        if len(instances) > 0:
            matched = instances.match_inds != -1
            if (~matched).any():
                self.init_tracks(frame_id, instances[~matched])
            if matched.any():
                self.update_tracks(frame_id, instances[matched])

        if not self.empty:
            self.update_state(frame_id)
            # Remove invalid tracks.
            self.remove_invalid_tracks()

    def track(self, frame_id: int, instances: "Instances") -> torch.Tensor:
        """ Tracking forward function.
        Assign a unique track id for each predicted instance in current frame by data association.

        Args:
            frame_id (int): The id of current frame, 0-index.
            instances (Instances): Instance predictions in current frame.

        Returns:
            ids (Tensor): Track ids for instances in current frame.
        """
        num_insts = len(instances)
        if num_insts == 0:
            self.update(frame_id, instances)
            return torch.arange(num_insts, device=instances.im_inds.device)

        instances = self.instantiate(instances)
        device = instances.ids.device

        if self.empty:
            instances.ids = torch.arange(
                self.num_tracks, self.num_tracks + num_insts, device=device
            )
        else:
            """ First association with confirmed tracks using embeddings"""
            # indices of confirmed tracks in the tracker
            track_inds = (self.tracks.state != TrackState.Tentative).nonzero().squeeze(1)
            if self.cfg.with_embeds and len(track_inds) > 0:
                track_embeds = self.tracks.embeds[track_inds]
                # 1 - cosine(a, b) = 0.5 * ||a - b||^2
                dists = torch.cdist(track_embeds, instances.embeds).square() * 0.5

                if self.cfg.with_motion and self.cfg.fuse_motion:
                    # inds = self.tracks.state == TrackState.Lost
                    # self.tracks.mean[track_inds & inds, 7] = 0
                    # motion prediction in the current frame
                    self.tracks, costs = self.kf.track(self.tracks, instances.xyah)
                    dists[costs[track_inds].isnan()] = dists.new_tensor(float("nan"))
                    dists = dists * self.cfg.sigma + costs[track_inds] * (1 - self.cfg.sigma)

                row, col = linear_sum_assignment(dists.cpu().numpy())
                match_inds = (dists[row, col] <= self.cfg.match_reid_thr).nonzero().squeeze(1).cpu()
                r, c = row[match_inds], col[match_inds]
                instances.match_inds[c] = track_inds[r]

            """ Second association using mask IoU"""
            # indices of tracks which were tracked till the previous frame but unmatched in the current frame
            track_inds = (self.tracks.state != TrackState.Lost).nonzero().squeeze(1)
            if self.cfg.with_miou:
                track_inds = track_inds[[ind not in instances.match_inds for ind in track_inds]]
                # indices of unmatched instances in the current frame
                inst_inds = (instances.match_inds == -1).nonzero().squeeze(1)

                if len(track_inds) > 0 and len(inst_inds) > 0:
                    track_masks = self.tracks.masks[track_inds]
                    dists = 1 - pairwise_miou(track_masks, instances.masks[inst_inds])

                    row, col = linear_sum_assignment(dists.cpu().numpy())
                    match_inds = (dists[row, col] <= 1 - self.cfg.match_miou_thr).nonzero().squeeze(1).cpu()
                    r, c = row[match_inds], inst_inds[col[match_inds]]
                    instances.match_inds[c] = track_inds[r]

            """ Third association using box IoU"""
            if self.cfg.with_biou:
                # indices of tracks which were tracked till the previous frame but unmatched in the current frame
                track_inds = track_inds[[ind not in instances.match_inds for ind in track_inds]]
                # indices of unmatched instances in the current frame
                inst_inds = (instances.match_inds == -1).nonzero().squeeze(1)

                if len(track_inds) > 0 and len(inst_inds) > 0:
                    track_boxes = self.tracks.boxes[track_inds]
                    dists = 1 - pairwise_biou(track_boxes, instances.boxes[inst_inds])

                    row, col = linear_sum_assignment(dists.cpu().numpy())
                    match_inds = (dists[row, col] <= 1 - self.cfg.match_biou_thr).nonzero().squeeze(1).cpu()
                    r, c = row[match_inds], inst_inds[col[match_inds]]
                    instances.match_inds[c] = track_inds[r]

            """ Associate track ids for instances matched in the current frame"""
            matched = instances.match_inds != -1
            instances.ids[matched] = self.ids[instances.match_inds[matched]]

            """ Create new tracks for instances unmatched in the current frame"""
            instances.ids[~matched] = torch.arange(
                self.num_tracks, self.num_tracks + (~matched).sum(), device=device
            )

        """ Update the tracker"""
        self.update(frame_id, instances)

        return instances.ids
