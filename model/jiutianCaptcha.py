import multiprocessing

import timm
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
from lightning import LightningModule


class jiutianNet(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = timm.create_model(
            'tv_resnet101',
            pretrained=True,
            num_classes=360
        )
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        img, label = batch
        embd = self(img)
        loss = self.criterion(embd, label)
        self.log("train_loss", loss)
        return loss

    def forward(self, x):
        x = self.backbone(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=Config.lr)
        return optimizer

    def train_dataloader(self):
        from ds.spdata import SPDATA
        dataset = SPDATA('../1', sep='_', transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ))

        return DataLoader(dataset, batch_size=8, num_workers=multiprocessing.cpu_count(),
                          shuffle=True, pin_memory=True, persistent_workers=True)


if __name__ == '__main__':
    from lightning import seed_everything
    from lightning import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from config import Config

    seed_everything(Config.seed)
    model = jiutianNet()

    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="train_loss", mode='min')

    trainer = Trainer(max_epochs=100, callbacks=[checkpoint_callback])
    trainer.fit(model)
