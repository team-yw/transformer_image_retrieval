import os
import multiprocessing


class Config:
    DATA_DIR = 'data'
    CSV_PATH = os.path.join(DATA_DIR, 'train_clean.csv')
    train_batch_size = 4
    val_batch_size = 10
    num_workers = multiprocessing.cpu_count()
    image_size = 224
    output_dim = 512
    hidden_dim = 1024
    input_dim = 3
    epochs = 35
    lr = 1e-3
    num_of_classes = 88313
    pretrained = True
    model_name = 'tv_resnet101'
    seed = 42
