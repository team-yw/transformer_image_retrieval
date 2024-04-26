import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAtrous(nn.Module):
    def __init__(self, in_channel, out_channel, size, dilation_rates=[3, 6, 9]):
        super().__init__()
        self.dilated_convs = [
            nn.Conv2d(in_channel, int(out_channel / 4),
                      kernel_size=3, dilation=rate, padding=rate)
            for rate in dilation_rates
        ]
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, int(out_channel / 4), kernel_size=1),
            nn.ReLU(),
            nn.Upsample(size=(size, size), mode='bilinear')
        )
        self.dilated_convs.append(self.gap_branch)
        self.dilated_convs = nn.ModuleList(self.dilated_convs)

    def forward(self, x):
        local_feat = []
        for dilated_conv in self.dilated_convs:
            local_feat.append(dilated_conv(x))
        local_feat = torch.cat(local_feat, dim=1)
        return local_feat


class DolgLocalBranch(nn.Module):
    def __init__(self, in_channel, out_channel, size, hidden_channel=2048):
        super().__init__()
        self.multi_atrous = MultiAtrous(in_channel, hidden_channel, size=int(size))
        self.conv1x1_1 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False)
        self.conv1x1_3 = nn.Conv2d(out_channel, out_channel, kernel_size=1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
        self.softplus = nn.Softplus()

    def forward(self, x):
        local_feat = self.multi_atrous(x)

        local_feat = self.conv1x1_1(local_feat)
        local_feat = self.relu(local_feat)
        local_feat = self.conv1x1_2(local_feat)
        local_feat = self.bn(local_feat)

        attention_map = self.relu(local_feat)
        attention_map = self.conv1x1_3(attention_map)
        attention_map = self.softplus(attention_map)

        local_feat = F.normalize(local_feat, p=2, dim=1)
        local_feat = local_feat * attention_map

        return local_feat


class OrthogonalFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, local_feat, global_feat):
        global_feat_norm = torch.norm(global_feat, p=2, dim=1)
        projection = torch.bmm(global_feat.unsqueeze(1), torch.flatten(
            local_feat, start_dim=2))
        projection = torch.bmm(global_feat.unsqueeze(
            2), projection).view(local_feat.size())
        projection = projection / \
                     (global_feat_norm * global_feat_norm).view(-1, 1, 1, 1)
        orthogonal_comp = local_feat - projection
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)
        return torch.cat([global_feat.expand(orthogonal_comp.size()), orthogonal_comp], dim=1)


class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super(ASPP, self).__init__()

        self.aspp = []
        self.aspp.append(nn.Conv2d(in_c, 512, 1, 1))

        for dilation in [6, 12, 18]:
            _padding = (dilation * 3 - dilation) // 2
            self.aspp.append(nn.Conv2d(in_c, 512, 3, 1, padding=_padding, dilation=dilation))
        self.aspp = nn.ModuleList(self.aspp)
        self.im_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(in_c, 512, 1, 1),
                                     nn.ReLU())
        conv_after_dim = 512 * (len(self.aspp) + 1)
        self.bn = nn.Sequential(
            nn.BatchNorm2d(conv_after_dim),
            nn.ReLU()
        )

        self.conv_after = nn.Sequential(
            nn.Conv2d(conv_after_dim, out_c, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_c * 2, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
        )

        self.conv1x1_1 = nn.Conv2d(out_c, out_c, kernel_size=1)
        self.sp = nn.ReLU()
        self.bn_2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        aspp_out = [F.interpolate(self.im_pool(x), scale_factor=(h, w), mode="bilinear", align_corners=False)]
        for i in range(len(self.aspp)):
            aspp_out.append(self.aspp[i](x))
        aspp_out = torch.cat(aspp_out, 1)
        aspp_out = self.bn(aspp_out)
        x = self.conv_after(aspp_out)

        attention_map = self.sp(x)
        attention_map = self.conv1x1_1(attention_map)
        attention_map = self.sp(attention_map)

        x = F.normalize(x)
        x = x * attention_map

        return x
