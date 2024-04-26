import random
import tarfile
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset
import os

from torchvision.transforms import transforms


class SPDATA(Dataset):
    seed = 27

    def __init__(self, labelPath, imgPath, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__()

        self.labelPath = labelPath
        self.imgPath = imgPath
        self.transform = transform
        self.target_transform = target_transform
        self.img = []

        labels = set()
        with open(self.labelPath) as f:
            lines = f.readlines()
        for i in lines:
            path, label = i.split(' ')
            labels.add(label[:-1])
            self.img.append((path, label[:-1]))
        labels = list(labels)
        labels.sort()
        self.labels = {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        path, label = self.img[idx]
        label = self.labels[label]
        img = Image.open(os.path.join(self.imgPath, path))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


