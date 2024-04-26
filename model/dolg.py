import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from ds.spdata import SPDATA

from torch.utils.data import DataLoader
from lightning import LightningModule

from config import Config
from model.gem_pool import GeM
from model.arcface import ArcFace


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
        self.conv1x1_2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=1, bias=False)
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


# 正交融合
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


class GlobalLocalFusion_without_att(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.local_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, local_feat, global_feat):
        local_feat = self.relu(self.local_conv(local_feat))
        batch_size, channels, height, width = local_feat.size()
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1).expand(batch_size, channels, height, width)

        feat = torch.cat([local_feat, global_feat], dim=1)
        feat = self.conv(feat)
        feat = self.norm(feat)

        return feat


class GlobalLocalFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.local_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.attention_conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()

    def forward(self, local_feat, global_feat):
        local_feat = self.local_conv(local_feat)

        batch_size, channels, height, width = local_feat.size()
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1).expand(batch_size, channels, height, width)

        feat = torch.cat([local_feat, global_feat], dim=1)

        attention_map = self.relu(feat)
        attention_map = self.attention_conv(attention_map)
        attention_map = self.softplus(attention_map)

        feat = F.normalize(feat, p=2, dim=1)

        feat = feat * attention_map
        feat = self.conv(feat)
        feat = self.norm(feat)

        return feat


class SENetAttentionFusion_9(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super().__init__()
        self.input_dim = input_dim
        self.reduction = reduction
        self.W_l = nn.Linear(input_dim, input_dim // reduction, bias=False)
        self.W_g = nn.Linear(input_dim, input_dim // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.V = nn.Linear(input_dim // reduction, input_dim, bias=False)

    def forward(self, local_feat, global_feat):
        # 计算局部特征和全局特征的SENet注意力权重
        local_embedding = self.relu(self.W_l(local_feat))
        global_embedding = self.relu(self.W_g(global_feat))
        # attention_weights = self.V(torch.cat([local_embedding, global_embedding], dim=1)).unsqueeze(1)
        # attention_weights = self.V(local_embedding + global_embedding).unsqueeze(1)
        attention_weights = self.V(local_embedding + global_embedding)
        attention_weights = torch.sigmoid(attention_weights)
        # print(local_feat.shape, global_feat.shape, attention_weights.shape)
        # 对局部特征和全局特征进行加权融合
        fused_feat = attention_weights * local_feat + (1 - attention_weights) * global_feat

        return fused_feat


class ronghe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.local_fc = torch.nn.Linear(input_dim, input_dim)
        self.global_fc = torch.nn.Linear(input_dim, input_dim)
        self.local_fc_att = torch.nn.Linear(input_dim, input_dim)
        self.global_fc_att = torch.nn.Linear(input_dim, input_dim)
        self.norm_l = nn.BatchNorm1d(input_dim)
        self.norm_g = nn.BatchNorm1d(input_dim)
        # self.norm = nn.BatchNorm2d(input_dim)

        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(input_dim * 2, Config.output_dim)

    def forward(self, local_feat, global_feat):
        local_feat = self.relu(self.norm_l(self.local_fc(local_feat)))
        global_feat = self.relu(self.norm_g(self.global_fc(global_feat)))
        att_lo = self.relu(self.local_fc_att(local_feat))
        att_go = self.relu(self.global_fc_att(global_feat))

        local_feat = local_feat * att_lo
        global_feat = global_feat * att_go
        feat = torch.cat([local_feat, global_feat], dim=1)
        feat = self.fc(feat)
        return feat


class TransformerFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformerFusion, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, local_feat, global_feat):
        # local_feat shape: [batch_size, local_feat_dim]
        # global_feat shape: [batch_size, global_feat_dim]
        x = torch.cat([local_feat.unsqueeze(1), global_feat.unsqueeze(1)], dim=1)  # [batch_size, 2, feat_dim]
        x = x.transpose(0, 1)  # [2, batch_size, feat_dim]
        x = self.transformer(x)  # [2, batch_size, feat_dim]
        x = x.mean(dim=0)  # [batch_size, feat_dim]
        x = self.fc(x)  # [batch_size, output_dim]
        return x


class VectorFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        hidden_dim = input_dim * 4
        # 调用父类的初始化方法
        super(VectorFusion, self).__init__()
        # 定义一个线性层，将输入向量映射到隐藏层
        self.linear1 = nn.Linear(input_dim * 2, hidden_dim)
        # 定义一个transformer层，使用多头自注意力机制
        self.transformer = nn.TransformerEncoderLayer(hidden_dim, nhead=4)
        # 定义一个线性层，将隐藏层映射到输出向量
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, local_feat, global_feat):
        # 将两个输入向量拼接起来，形成一个二维张量
        x = torch.cat([local_feat, global_feat], dim=1)
        # 通过第一个线性层，得到隐藏层表示
        h = self.linear1(x)
        # 通过transformer层，得到融合后的表示
        h = self.transformer(h)
        # 通过第二个线性层，得到输出向量
        y = self.linear2(h)
        # 返回输出向量的第一行，即融合后的向量
        return y


# 输入两个向量local_feat和global_feat,维度为[batch_size, input_dim],输出融合后的向量，维度为[batch_size, output_dim],使用transformer方法对这两个向量进行融合，可以加一些你认为更优的小技巧


class DolgNet(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, num_of_classes):
        super().__init__()
        self.cnn = timm.create_model(
            Config.model_name,
            pretrained=True,
            features_only=True,
            in_chans=input_dim,
            out_indices=(2, 3),

        )

        self.orthogonal_fusion = VectorFusion(Config.hidden_dim, Config.output_dim)
        # self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = DolgLocalBranch(512, hidden_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gem_pool = GeM()
        # self.GlobalLocalFusion = GlobalLocalFusion(Config.hidden_dim)
        # self.SENetAttentionFusion = SENetAttentionFusion(Config.hidden_dim)
        # 将Linear改为1*1的卷积层
        # self.fc_1 = nn.Linear(1024, hidden_dim)
        # self.fc_2 = nn.Linear(int(2 * hidden_dim), output_dim)
        self.fc_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.fc_2 = nn.Conv2d(int(hidden_dim), hidden_dim, kernel_size=1)
        # self.fc_3 = nn.Conv2d(int(hidden_dim), output_dim, kernel_size=1)
        # self.fc_3 = nn.Linear(int(hidden_dim), output_dim)
        # self.fc_2 = nn.Conv2d(int(2 * hidden_dim), output_dim, kernel_size=1)

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
        local_feat = self.fc_2(self.gem_pool(local_feat))  # ,1024
        local_feat = local_feat.view(local_feat.size(0), -1)

        global_feat = self.fc_1(self.gem_pool(output[1]))  # ,1024
        global_feat = global_feat.view(global_feat.size(0), -1)
        feat = self.orthogonal_fusion(local_feat, global_feat)

        # feat = self.fc_3(feat)
        # feat = feat.view(feat.size(0), -1)

        return feat

    def training_step(self, batch, batch_idx):
        img, label = batch
        embd = self(img)
        loss, logits = self.criterion(embd, label)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.SGD(self.parameters(), lr=self.lr,
        #                       momentum=0.9, weight_decay=1e-5)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=1000)
        return optimizer
        # return [optimizer], [scheduler]

    def train_dataloader(self):
        # oxf5k
        dataset = SPDATA(r"data/oxbuild_images/", '_0', transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([32] * 2, antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))

        return DataLoader(dataset, batch_size=Config.train_batch_size, num_workers=Config.num_workers,
                          shuffle=True, pin_memory=True, persistent_workers=True)
