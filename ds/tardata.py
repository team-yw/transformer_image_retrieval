import random
import tarfile
from typing import Optional, Callable
from PIL import Image
from torch.utils.data import Dataset, get_worker_info, DataLoader
from torchvision.transforms import transforms


class SPDATA(Dataset):
    seed = 27

    def __init__(self, tar_path, sep, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 tv_rate=0.8, train: bool = True):
        super().__init__()

        self.tar_obj = {}
        self.sep = sep
        self.tar_path = tar_path
        self.transform = transform
        self.target_transform = target_transform
        self.tv_rate = tv_rate
        self.train = train
        # 打开Tar文件
        with tarfile.open(self.tar_path) as tar:
            # 获取Tar文件中的所有文件名
            self.names = [x.name for x in tar.getmembers()]
        # 生成标签列表
        labels = set([x.split(self.sep)[0] for x in self.names])
        labels = list(labels)
        labels.sort()
        self.labels = {label: idx for idx, label in enumerate(labels)}
        # 打乱数据集
        random.Random(self.seed).shuffle(self.names)
        # 划分训练集和验证集的文件名
        if train:
            self.names = self.names[:int(len(self.names) * tv_rate)]
        else:
            self.names = self.names[int(len(self.names) * tv_rate):]

    def __len__(self):
        return len(self.names)

    def label2index(self):
        return self.labels

    def index2label(self):
        lb = {}
        for k, v in self.labels.items():
            lb[v] = k
        return lb

    def __getitem__(self, idx):
        # 获取文件名
        name = self.names[idx]
        # 获得文件io
        img = self.read_file(name)
        # 获取标签
        label = name.split(self.sep)[0]
        # 获取标签索引
        label = self.labels[label]
        # 将文件io转换为PIL.Image.Image
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def read_file(self, file_name):
        worker = get_worker_info()
        worker = worker.id if worker else None

        if worker not in self.tar_obj:
            # self.tar_obj[worker][0]为tarfile
            # self.tar_obj[worker][1]为{文件名: 地址}
            self.tar_obj[worker] = [tarfile.open(self.tar_path)]
            members = self.tar_obj[worker][0].getmembers()
            self.tar_obj[worker].append({m.name: m for m in members})
        return self.tar_obj[worker][0].extractfile(self.tar_obj[worker][1][file_name])
