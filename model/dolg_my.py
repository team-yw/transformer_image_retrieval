import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import Config
from model.gem_pool import GeM
from model.arcface import ArcFace
from model.base import base_net


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
    def __init__(self, in_channel, out_channel, hidden_channel=2048):
        super().__init__()
        self.multi_atrous = MultiAtrous(in_channel, hidden_channel, size=int(Config.image_size / 8))
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


class my_Fusion(nn.Module):
    def __init__(self, num_layers, num_heads):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=1808, nhead=self.num_heads)
            for _ in range(self.num_layers)
        ])

    def forward(self, local_feat, global_feat):
        # print(local_feat.shape, global_feat.shape)
        local_feat = torch.flatten(local_feat, start_dim=2)
        projection = torch.bmm(global_feat.unsqueeze(1), local_feat)
        projection = torch.flatten(projection, start_dim=1)
        # print(projection.shape)
        projection = torch.cat([projection, global_feat], dim=1)
        for transformer_block in self.transformer_blocks:
            projection = transformer_block(projection)
        return projection


class DolgNet(base_net):
    def __init__(self, input_dim, hidden_dim, output_dim, num_of_classes):
        super().__init__()
        self.cnn = timm.create_model(
            'resnet101.tv_in1k',
            # 'swin_base_patch4_window7_224',
            pretrained=True,
            features_only=True,
            in_chans=input_dim,
            # drop_rate=0.5,
            out_indices=(2, 3)
        )
        self.classes_num = num_of_classes

        self.validation_step_outputs = []
        self.validation_label = []
        # self.orthogonal_fusion = OrthogonalFusion()
        self.orthogonal_fusion = my_Fusion(num_layers=10, num_heads=8)
        self.local_branch = DolgLocalBranch(512, hidden_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gem_pool = GeM()
        self.fc_1 = nn.Linear(1024, hidden_dim)
        # self.fc_2 = nn.Linear(int(2 * hidden_dim), output_dim)
        self.fc_2 = nn.Linear(1808, output_dim)

        self.criterion = ArcFace(
            in_features=output_dim,
            out_features=num_of_classes,
            scale_factor=30,
            margin=0.15,
            criterion=nn.CrossEntropyLoss()
        )
        self.lr = Config.lr

    def forward(self, x):
        output = self.cnn(x)

        local_feat = self.local_branch(output[0])  # ,hidden_channel,16,16
        global_feat = self.fc_1(self.gem_pool(output[1]).squeeze())  # ,1024
        feat = self.orthogonal_fusion(local_feat, global_feat)
        # feat = self.gap(feat).squeeze()
        feat = self.fc_2(feat)

        return feat

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr,
                              momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    # input_dim, hidden_dim, output_dim, num_of_classes
    model = DolgNet(input_dim=3, hidden_dim=1024, output_dim=512, num_of_classes=16)
    batch = torch.randn(2, 3, 224, 224)
    out = model(batch)
    print(out.shape)
