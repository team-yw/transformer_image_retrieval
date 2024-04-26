import torch
from torch import nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes: int = 10, feat_dim: int = 2, clamp: int = 1e-12):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.clamp = clamp

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        centers = self.centers.t()
        distmat = x.square().sum(dim=1, keepdim=True) + centers.square().sum(dim=0, keepdim=True)
        # B F @ B C -> Gather C -> B F @ F
        distmat = distmat - 2 * x @ centers
        dist = torch.gather(distmat, 1, labels.view(-1, 1))
        return dist.clamp(min=self.clamp, max=1 / self.clamp).mean()
