# oxf5k:https://thor.robots.ox.ac.uk/datasets/oxford-buildings/oxbuild_images.tgz
import random
import tarfile
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset
import os

from torchvision.transforms import transforms


class SPDATA_tar(Dataset):
    seed = 27

    def __init__(self, dataTarFile, sep, transform: Optional[Callable] = None, tv_rate=0.8, train: bool = True):
        # dataTarFile是一个.tgz文化的路径，如：C:\Users\shuke\Downloads\oxbuild_images.tgz
        # 将文件名按照sep来分割，区sep[0]为标签，如：oxc1_000001.jpg，sep为'_'，则标签为oxc1
        # tv_rate是训练集和验证集的比例，如：0.8，表示训练集占80%，验证集占20%
        super().__init__()

        self.sep = sep
        self.dataTarFile = dataTarFile
        self.transform = transform
        # 打开Tar文件
        self.tar = tarfile.open(self.dataTarFile)
        # 获取Tar文件中的所有文件名和文件地址 name:address
        self.members = [(x.name, x) for x in self.tar.getmembers()]
        # 生成标签列表
        labels = set([x[0].split(self.sep)[0] for x in self.members])
        labels = list(labels)
        labels.sort()
        self.labels = {label: i for i, label in enumerate(labels)}
        # 打乱数据集
        random.Random(self.seed).shuffle(self.members)
        # 划分训练集和验证集
        if train:
            self.members = self.members[:int(len(self.members) * tv_rate)]
        else:
            self.members = self.members[int(len(self.members) * tv_rate):]

    def __len__(self):
        return len(self.members)

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        name, addr = self.members[idx]
        label = name.split(self.sep)[0]
        label = self.labels[label]
        img = Image.open(self.tar.extractfile(addr))
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class SPDATA(Dataset):
    seed = 27

    def __init__(self, fileDir, sep, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 tv_rate=0.8, train: bool = True):
        super().__init__()

        self.fileDir = fileDir
        self.files = os.listdir(fileDir)
        self.sep = sep
        self.transform = transform
        self.target_transform = target_transform

        # 生成标签列表
        labels = set([x.split(self.sep)[0] for x in self.files])
        labels = list(labels)
        labels.sort()
        self.labels = {label: i for i, label in enumerate(labels)}

        # 打乱数据集
        random.Random(self.seed).shuffle(self.files)
        # 划分训练集和验证集
        if train:
            self.files = self.files[:int(len(self.files) * tv_rate)]
        else:
            self.files = self.files[int(len(self.files) * tv_rate):]

    def __len__(self):
        return len(self.files)

    def label2index(self):
        return self.labels

    def index2label(self):
        lb = {}
        for k, v in self.labels.items():
            lb[v] = k
        return lb

    def __getitem__(self, idx):
        path, label = self.files[idx], self.files[idx].split(self.sep)[0]
        label = self.labels[label]
        img = Image.open(os.path.join(self.fileDir, path))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


