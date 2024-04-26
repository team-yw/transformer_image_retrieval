import torch
import torch.nn as nn


class SEblock(nn.Module):
    def __init__(self, channel, r=0.5):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        # 全连接得到权重
        weight = self.fc(x)
        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))
        scale = weight * x
        return scale


class SEblock_2D(nn.Module):
    def __init__(self, channel, r=0.5):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)
        weight = self.fc(branch)
        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))
        scale = weight * x

        return scale
