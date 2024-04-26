import multiprocessing
import platform

import requests
from lightning import seed_everything
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from ds.tardata import SPDATA
import torchvision
from reretry import retry
from model.my import MyNet




@retry(exceptions=requests.exceptions.RequestException)
def main():
    windows = platform.system() == 'Windows'

    datalocation = './data' if windows else '/dev/shm'
    batch_size = 4 if windows else 32
    num_workers = 1 if windows else min(multiprocessing.cpu_count(), 32)
    # -----init-----

    seed_everything(27)
    model = MyNet(features_num=1024, image_size=224, classes_num=16)

    tsf = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize([256, 256]),
            torchvision.transforms.CenterCrop([224, 224]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.7773, 0.7435, 0.7320],
                                             std=[0.2623, 0.2786, 0.2862])])

    train_set = SPDATA(rf"{datalocation}/inshop_all_224.tar", sep='#', train=True, transform=tsf)

    val_set = SPDATA(rf"{datalocation}/inshop_all_224.tar", sep='#', train=False, transform=tsf)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, persistent_workers=True, drop_last=True)

    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="val_acc_10", mode='max')
    trainer = Trainer(max_epochs=200, callbacks=[checkpoint_callback], num_sanity_val_steps=20)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
