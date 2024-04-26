import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, requires_grad=False):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.tensor(p), requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.pow(1.0 / self.p)
        return x

    def extra_repr(self):
        return f"p={self.p.item():.4f}, eps={self.eps}"
