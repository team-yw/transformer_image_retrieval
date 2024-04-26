import random
from collections import defaultdict
import torch.nn.functional as F

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from numpy.linalg import norm
import numpy as np
from timm.layers import SelectAdaptivePool2d
from model.arcface import ArcFace
from sklearn.cluster import KMeans
from model.DolgClass import DolgLocalBranch

from model.base import base_net


class multiFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 512

    def forward(self, x):
        return x


class feature_extractor(nn.Module):
    def __init__(self, example, features_num):
        super().__init__()
        self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=True)

        num_of_features = sum([x.size(-1) for x in example]) if example[0].size(-1) != example[0].size(-2) else sum(
            [x.size(1) for x in example])

        self.d1 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            timm.create_model('vit_tiny_patch16_224', pretrained=True, img_size=112, num_classes=256)
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1),
            nn.Upsample(size=(64, 64), mode='bilinear'),
            timm.create_model('vit_tiny_patch16_224', pretrained=True, img_size=64, num_classes=256)
        )

        self.d3 = nn.Sequential(
            nn.Conv2d(512, 3, kernel_size=1),
            nn.Upsample(size=(32, 32), mode='bilinear'),
            timm.create_model('vit_tiny_patch16_224', pretrained=True, img_size=32, num_classes=256)
        )

        self.d4 = nn.Sequential(
            nn.Conv2d(1024, 3, kernel_size=1),
            nn.Upsample(size=(16, 16), mode='bilinear'),
            timm.create_model('vit_tiny_patch16_224', pretrained=True, img_size=16, num_classes=256)
        )

        self.d5 = nn.Sequential(
            nn.Conv2d(2048, 3, kernel_size=1),
            nn.Upsample(size=(16, 16), mode='bilinear'),
            timm.create_model('vit_tiny_patch16_224', pretrained=True, img_size=16, num_classes=256)
        )

        self.tan = nn.Tanh()

        self.rl = nn.ReLU()

        self.l1 = nn.Linear(1280, 1280 * 4)
        self.l2 = nn.Linear(1280 * 4, features_num)

        self.att = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(5, 5, kernel_size=1),
            nn.Softplus()
        )

    def adjust_channel(self, x):
        if x.shape[-1] != x.shape[-2]:
            x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        for i, v in enumerate(x):
            # 调整图片的通道数，让通道在第一维
            x[i] = self.adjust_channel(x[i])
            # # gem提取图片特征
        # x1 = self.global_pool(self.d1(x[0]))
        x1 = self.d1(x[0])
        # print('x1', x1.shape)
        # x2 = self.global_pool(self.d2(x[1]))
        x2 = self.d2(x[1])
        # print(x2.shape)
        x3 = self.d3(x[2])
        # print(x3.shape)
        # x4 = self.global_pool(self.d4(x[3]))
        x4 = self.d4(x[3])
        # print(x4.shape)
        # x5 = self.global_pool(self.d5(x[4]))
        x5 = self.d5(x[4])
        # print(x5.shape)

        x = torch.stack((x1, x2, x3, x4, x5), dim=1)
        x = x.view(-1, 5, 16, 16)
        # print(x.shape)
        att = self.att(x)
        x = x * att
        # print('x', x.shape)
        x = self.tan(x)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = self.l1(x)
        x = self.rl(x)
        x = self.l2(x)

        # for i in x:
        #     print(i.shape)

        return x


class MyNet(base_net):
    def __init__(self, features_num, image_size, classes_num):
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.validation_label = []
        self.classes_num = classes_num
        self.backbone = timm.create_model(
            # 'swin_base_patch4_window7_224',
            'resnet101.tv_in1k',
            pretrained=True,
            # drop_rate=0.5,
            features_only=True,
            # num_classes=features_num,
        )

        self.criterion1 = ArcFace(
            in_features=features_num,
            out_features=self.classes_num,
            scale_factor=30,
            margin=0.15,
            criterion=nn.CrossEntropyLoss()
        )
        self.criterion2 = nn.TripletMarginLoss()
        # 随机生成数据，计算网络数据
        example = self.backbone(torch.randn(1, 3, image_size, image_size))
        num_of_features = len(example)

        # 特征融合
        self.feature_extractor = feature_extractor(example, features_num)

    def forward(self, x):
        out = self.backbone(x)
        # for i in out:
        #     print(i.shape)

        out = self.feature_extractor(out)

        return out

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3,
                              momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1000)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        anc, pos, neg = batch[0]
        anc_l, pos_l, neg_l = batch[1]

        anc = self(anc)
        pos = self(pos)
        neg = self(neg)
        loss1 = self.criterion2(anc, pos, neg) * 30
        loss2 = torch.mean(
            torch.stack([self.criterion1(anc, anc_l), self.criterion1(pos, pos_l), self.criterion1(neg, neg_l)]))
        loss = loss2 + loss1
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.validation_step_outputs = []
            self.validation_label = []
        img, label = batch

        embd = self(img)
        self.validation_step_outputs.extend(embd.cpu().detach().numpy())

        label = label.tolist()
        self.validation_label.extend(label)


if __name__ == '__main__':
    model = MyNet(features_num=512, image_size=224, classes_num=16)
    batch = torch.randn(2, 3, 224, 224)
    out = model(batch)
    print(out.shape)
