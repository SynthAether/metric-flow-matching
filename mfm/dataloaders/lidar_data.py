import torch
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import laspy
import numpy as np
from scipy.spatial import cKDTree
import math
from functools import partial


### code adapted from https://github.com/facebookresearch/generalized-schrodinger-bridge-matching.git
class GaussianMM:
    def __init__(self, mu, var):
        super().__init__()
        self.centers = torch.tensor(mu)
        self.logstd = torch.tensor(var).log() / 2.0
        self.K = self.centers.shape[0]

    def logprob(self, x):
        logprobs = self.normal_logprob(
            x.unsqueeze(1), self.centers.unsqueeze(0), self.logstd
        )
        logprobs = torch.sum(logprobs, dim=2)
        return torch.logsumexp(logprobs, dim=1) - math.log(self.K)

    def normal_logprob(self, z, mean, log_std):
        mean = mean + torch.tensor(0.0)
        log_std = log_std + torch.tensor(0.0)
        c = torch.tensor([math.log(2 * math.pi)]).to(z)
        inv_sigma = torch.exp(-log_std)
        tmp = (z - mean) * inv_sigma
        return -0.5 * (tmp * tmp + 2 * log_std + c)

    def __call__(self, n_samples):
        idx = torch.randint(self.K, (n_samples,)).to(self.centers.device)
        mean = self.centers[idx]
        return torch.randn(*mean.shape).to(mean) * torch.exp(self.logstd) + mean


class LidarDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.max_dim = args.dim
        self.whiten = args.whiten
        self.p0_mu = [
            [-4, -2, 0.5],
            [-3.75, -1.125, 0.5],
            [-3.5, -0.25, 0.5],
            [-3.25, 0.675, 0.5],
            [-3, 1.5, 0.5],
        ]
        self.p0_var = 0.02
        self.p1_mu = [[2, -2, 0.5], [2.6, -1.25, 0.5], [3.2, -0.5, 0.5]]
        self.p1_var = 0.03
        self.k = 20
        self.n_samples = 5000
        self.num_timesteps = 2
        self.split_ratios = args.split_ratios
        self._prepare_data()

    def _prepare_data(self):
        las = laspy.read(self.data_path)
        # Extract only "ground" points.
        self.mask = las.classification == 2
        # Original Preprocessing
        x_offset, x_scale = las.header.offsets[0], las.header.scales[0]
        y_offset, y_scale = las.header.offsets[1], las.header.scales[1]
        z_offset, z_scale = las.header.offsets[2], las.header.scales[2]
        dataset = np.vstack(
            (
                las.X[self.mask] * x_scale + x_offset,
                las.Y[self.mask] * y_scale + y_offset,
                las.Z[self.mask] * z_scale + z_offset,
            )
        ).transpose()
        mi = dataset.min(axis=0, keepdims=True)
        ma = dataset.max(axis=0, keepdims=True)
        dataset = (dataset - mi) / (ma - mi) * [10.0, 10.0, 2.0] + [-5.0, -5.0, 0.0]

        self.dataset = torch.tensor(dataset, dtype=torch.float32)
        self.tree = cKDTree(dataset)

        x0_gaussian = GaussianMM(self.p0_mu, self.p0_var)(self.n_samples)
        x1_gaussian = GaussianMM(self.p1_mu, self.p1_var)(self.n_samples)

        x0 = self.get_tangent_proj(x0_gaussian)(x0_gaussian)
        x1 = self.get_tangent_proj(x1_gaussian)(x1_gaussian)

        split_index = int(self.n_samples * self.split_ratios[0])

        self.scaler = StandardScaler()
        if self.whiten:
            self.dataset = torch.tensor(
                self.scaler.fit_transform(dataset), dtype=torch.float32
            )
            x0 = torch.tensor(self.scaler.transform(x0), dtype=torch.float32)
            x1 = torch.tensor(self.scaler.transform(x1), dtype=torch.float32)

        train_x0 = x0[:split_index]
        val_x0 = x0[split_index:]
        train_x1 = x1[:split_index]
        val_x1 = x1[split_index:]
        self.val_x0 = val_x0

        # Adjust split_index to ensure minimum validation samples
        if self.n_samples - split_index < self.batch_size:
            split_index = self.n_samples - self.batch_size

        self.train_dataloaders = [
            DataLoader(
                train_x0,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            ),
            DataLoader(
                train_x1,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            ),
        ]
        self.val_dataloaders = [
            DataLoader(
                val_x0,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
            ),
            DataLoader(
                val_x1,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            ),
        ]
        self.test_dataloaders = [
            DataLoader(
                self.val_x0,
                batch_size=self.val_x0.shape[0],
                shuffle=False,
                drop_last=False,
            ),
            DataLoader(
                self.dataset,
                batch_size=self.dataset.shape[0],
                shuffle=False,
                drop_last=False,
            ),
        ]

        self.metric_samples_dataloaders = [
            DataLoader(
                self.dataset[: dataset.shape[0] // 2],
                batch_size=dataset.shape[0] // 2,  # balanced batches
                shuffle=False,
                drop_last=False,
            ),
            DataLoader(
                self.dataset[dataset.shape[0] // 2 :],
                batch_size=dataset.shape[0] // 2,  # balanced batches
                shuffle=False,
                drop_last=False,
            ),
        ]

    def train_dataloader(self):
        combined_loaders = {
            "train_samples": CombinedLoader(self.train_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        combined_loaders = {
            "val_samples": CombinedLoader(self.val_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }

        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def test_dataloader(self):
        return CombinedLoader(self.test_dataloaders)

    def get_tangent_proj(self, points):
        w = self.get_tangent_plane(points)
        return partial(LidarDataModule.projection_op, w=w)

    def get_tangent_plane(self, points, temp=1e-3):
        points_np = points.detach().cpu().numpy()
        _, idx = self.tree.query(points_np, k=self.k)
        nearest_pts = self.dataset[idx]
        nearest_pts = torch.tensor(nearest_pts).to(points)

        dists = (points.unsqueeze(1) - nearest_pts).pow(2).sum(-1, keepdim=True)
        weights = torch.exp(-dists / temp)

        # Fits plane with least vertical distance.
        w = LidarDataModule.fit_plane(nearest_pts, weights)
        return w

    @staticmethod
    def fit_plane(points, weights=None):
        """Expects points to be of shape (..., 3).
        Returns [a, b, c] such that the plane is defined as
            ax + by + c = z
        """
        D = torch.cat([points[..., :2], torch.ones_like(points[..., 2:3])], dim=-1)
        z = points[..., 2]
        if weights is not None:
            Dtrans = D.transpose(-1, -2)
        else:
            DW = D * weights
            Dtrans = DW.transpose(-1, -2)
        w = torch.linalg.solve(
            torch.matmul(Dtrans, D), torch.matmul(Dtrans, z.unsqueeze(-1))
        ).squeeze(-1)
        return w

    @staticmethod
    def projection_op(x, w):
        """Projects points to a plane defined by w."""
        # Normal vector to the tangent plane.
        n = torch.cat([w[..., :2], -torch.ones_like(w[..., 2:3])], dim=1)

        pn = torch.sum(x * n, dim=-1, keepdim=True)
        nn = torch.sum(n * n, dim=-1, keepdim=True)

        # Offset.
        d = w[..., 2:3]

        # Projection of x onto n.
        projn_x = ((pn + d) / nn) * n

        # Remove component in the normal direction.
        return x - projn_x
