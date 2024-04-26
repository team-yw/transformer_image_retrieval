import torch
import torch.nn as nn
import torch.nn.functional as F


# softmax loss
class SoftmaxLoss(nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()

    def forward(self, feat, target):
        return F.cross_entropy(feat, target)


# center loss
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lamda):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.lamda = lamda

    def forward(self, feat, target):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        centers_batch = self.centers.index_select(0, target)
        diff = centers_batch - feat
        loss = self.lamda * torch.sum(diff ** 2) / 2.0 / batch_size
        return loss


# softmax + center loss
class SoftmaxCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lamda):
        super(SoftmaxCenterLoss, self).__init__()
        self.softmaxLoss = SoftmaxLoss()
        self.centerLoss = CenterLoss(num_classes, feat_dim, lamda)

    def forward(self, feat, target):
        softmax_loss = self.softmaxLoss(feat, target)
        center_loss = self.centerLoss(feat, target)
        return softmax_loss + center_loss
