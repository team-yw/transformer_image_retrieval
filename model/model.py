import timm
import torch
import torch.nn as nn
import torch.optim as optim
from timm.layers import SelectAdaptivePool2d
from model.arcface import ArcFace
from model.base import base_net
from model.gem_pool import GeM


class feature_extractor(nn.Module):
    def __init__(self, features_num):
        super().__init__()
        self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=True)

        self.d1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 1),
            GeM()
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 1),
            GeM()
        )

        self.d3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 1),
            GeM()
        )

        self.d4 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 1),
            GeM()
        )

        self.d5 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 1),
            GeM()
        )

        self.tan = nn.Tanh()

        self.rl = nn.ReLU()

        self.l1 = nn.Linear(256 * 4, 1024 * 4)
        self.l2 = nn.Linear(1024 * 4, features_num)

        self.att = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=1),
            nn.Softplus()  # Sigmoid
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
        # x5 = self.d5(x[4])
        # print(x5.shape)

        # x = torch.stack((x1, x2, x3, x4, x5), dim=1)
        x = torch.stack((x1, x2, x3, x4), dim=1)
        x = x.view(-1, 4, 16, 16)
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
            'swin_base_patch4_window7_224',
            # 'resnet101',
            pretrained=True,
            drop_rate=0.5,
            features_only=True,
        )
        self.criterion = ArcFace(
            in_features=features_num,
            out_features=self.classes_num,
            scale_factor=30,
            margin=0.05,
            criterion=nn.CrossEntropyLoss()
        )

        # 特征融合
        self.feature_extractor = feature_extractor(features_num)

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


if __name__ == '__main__':
    model = MyNet(features_num=1024, image_size=224, classes_num=16)
    batch = torch.randn(2, 3, 224, 224)
    out = model(batch)
    print(out.shape)
