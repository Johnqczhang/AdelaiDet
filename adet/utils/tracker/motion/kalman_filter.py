import torch

# from mmtrack.models import MOTION


CHI2INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


# @MOTION.register_module()
class KalmanFilter(object):
    """Vectorized (based on torch.Tensor) version of a simple Kalman Filter for tracking bounding boxes in image space.

    The implementation is referred to https://github.com/nwojke/deep_sort.
    """

    def __init__(self, device="cpu", center_only=False):
        self.center_only = center_only
        if center_only:
            self.gating_threshold = CHI2INV95[2]
        else:
            self.gating_threshold = CHI2INV95[4]

        ndim, dt = 4, 1.
        self.ndim = ndim

        # Create Kalman filter model matrices.
        self._motion_mat = torch.eye(2 * ndim, device=device)
        self._motion_mat[:ndim, ndim:] += (torch.eye(ndim, device=device) * dt)
        self._update_mat = torch.eye(ndim, 2 * ndim, device=device)

        # Motion and observation uncertainty are chosen relative to the current state estimate.
        # These weights control the amount of uncertainty in the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, boxes):
        """Create track from unassociated boxes.

        Args:
            boxes (Tensor[float]): The Nx4 matrix of bounding boxes. Each row is `(x, y, a, h)`
                with center position `(x, y)`, aspect ratio `a`, and bounding box height `h`.

        Returns:
             (Tensor[float], Tensor[float]): Returns the state's mean matrix (Nx8) and
                covariance matrix (Nx8x8) of the new track.
                Unobserved velocities are initialized to 0 mean.
        """
        mean = boxes.new_zeros((boxes.size(0), 8))
        mean[:, :4] = boxes

        std = torch.ones_like(mean) * mean[:, 3:4]
        std[:, :4] *= (2 * self._std_weight_position)
        std[:, 2] = 1e-2
        std[:, 4:] *= (10 * self._std_weight_velocity)
        std[:, 6] = 1e-5
        covariance = torch.stack([torch.diag(s) for s in std.square()], dim=0)

        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Args:
            mean (Tensor[float]): The Nx8 mean matrix of the object states
                at the previous time step.
            covariance (Tensor[float]): The Nx8x8 covariance matrix of the object states
                at the previous time step.

        Returns:
            (Tensor[float], Tensor[float]): Returns the mean matrix (Nx8) and
                the covariance matrix (Nx8x8) of the predicted states.
                Unobserved velocities are initialized to 0 mean.
        """
        std = torch.ones_like(mean) * mean[:, 3:4]
        std[:, :4] *= self._std_weight_position
        std[:, 2] = 0.01
        std[:, 4:] *= self._std_weight_velocity
        std[:, 6] = 1e-5
        motion_cov = torch.stack([torch.diag(s) for s in std.square()], dim=0)
        covariance = self._motion_mat.matmul(covariance).matmul(self._motion_mat.t()) + motion_cov
        mean = mean.matmul(self._motion_mat.t())

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Args:
            mean (Tensor[float]): The state's mean matrix (Nx8).
            covariance (Tensor[float]): The state's covariance matrix (Nx8x8).

        Returns:
            (Tensor[float], Tensor[float]): Returns the projected mean and covariance
                matrix of the given state estimate.
        """
        std = torch.ones_like(mean[:, :self.ndim]) * mean[:, 3:4] * self._std_weight_position
        std[:, 2] = 0.1
        inno_cov = torch.stack([torch.diag(s) for s in std.square()], dim=0)
        covariance = self._update_mat.matmul(covariance).matmul(self._update_mat.t()) + inno_cov
        mean = mean.matmul(self._update_mat.t())

        return mean, covariance

    def update(self, mean, covariance, boxes):
        """Run Kalman filter correction step.

        Args:
            mean (Tensor[float]): The predicted state's mean matrix (Nx8).
            covariance (Tensor[float]): The state's covariance matrix (Nx8x8).
            boxes (Tensor[float]): The Nx4 matrix of bounding boxes. Each row is `(x, y, a, h)`
                with center position `(x, y)`, aspect ratio `a`, and bounding box height `h`.

        Returns:
             (Tensor[float], Tensor[float]): Returns the box-corrected state distribution.
        """
        proj_mean, proj_cov = self.project(mean, covariance)
        kalman_gain = self._update_mat.matmul(covariance).cholesky_solve(
            proj_cov.cholesky(upper=False), upper=False
        )
        mean += (boxes - proj_mean)[:, None].matmul(kalman_gain).squeeze(1)
        covariance -= kalman_gain.permute(0, 2, 1).matmul(proj_cov).matmul(kalman_gain)

        return mean, covariance

    def gating_distance(self, mean, covariance, boxes, only_position=False, metric="maha"):
        """Compute gating distance between state distribution and boxes.

        A suitable distance threshold can be obtained from `chi2inv95`.
        If `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Args:
            mean (Tensor[float]): Mean matrix over the state distribution (Mx8).
            covariance (Tensor[float]): Covariance of the state distribution (Mx8x8).
            boxes (Tensor[float]): The Nx4 matrix of bounding boxes. Each row is `(x, y, a, h)`
                with center position `(x, y)`, aspect ratio `a`, and bounding box height `h`.
            only_position (bool, optional): If True, distance computation is
                done with respect to the bounding box center position only.
                Defaults to False.
            metric (str): Defaults to Mahalanobis distance.

        Returns:
            Tensor[float]: Returns a MxN matrix, where the j-th element of the i-th row
            contains the squared Mahalanobis distance between the i-th
            (mean, covariance) and `boxes[j]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:, :2], covariance[:, :2, :2]
            boxes = boxes[:, :2]

        dists = -(mean[:, None] - boxes[None])  # (M, N, ndim)
        if metric == "gaussian":
            return dists.square().sum(dim=-1)
        elif metric == "maha":
            dists = dists.permute(0, 2, 1)  # (M, ndim, N)
            dists = dists.triangular_solve(
                covariance.cholesky(), upper=False
            )[0].square().sum(1)
            return dists
        else:
            raise ValueError(f"Invalid distance metric: {metric}")

    def track(self, tracks, bboxes):
        """Track forward.

        Args:
            tracks (Tracks): Track set with M tracks.
            bboxes (Tensor): Detected bounding boxes (Nx4).

        Returns:
            (Tracks, Tensor): Updated tracks and the cost matrix (MxN).
        """
        tracks.mean, tracks.covariance = self.predict(
            tracks.mean, tracks.covariance
        )
        costs = self.gating_distance(
            tracks.mean, tracks.covariance, bboxes, self.center_only
        )
        costs[costs > self.gating_threshold] = costs.new_tensor(float("nan"))

        return tracks, costs
