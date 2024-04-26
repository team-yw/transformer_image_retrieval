import datetime
import gc
import os
import random
from collections import defaultdict

import sklearn
import torch
import torchvision.transforms
from lightning import LightningModule
from numpy.linalg import norm
import numpy as np
from sklearn.cluster import KMeans, BisectingKMeans, FeatureAgglomeration, MiniBatchKMeans, SpectralBiclustering, \
    SpectralCoclustering
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F


def l2_norm(fea):
    """l2 norm feature get cos dis"""
    if fea is None:
        return fea
    if len(fea.shape) > 1:
        fea = fea / np.linalg.norm(fea, axis=1)[:, None]
    else:
        fea = fea / np.linalg.norm(fea)
    return fea


def sca(original_tensor, scale_factor):
    scaled_tensor = F.interpolate(original_tensor.unsqueeze(0).unsqueeze(0),
                                  scale_factor=(1, scale_factor, scale_factor), mode='nearest')
    restored_tensor = F.interpolate(scaled_tensor, scale_factor=(1, 1 / scale_factor, 1 / scale_factor),
                                    mode='nearest').squeeze(0).squeeze(0)
    return restored_tensor


def sca_image(original_tensor, scale_factor):
    tsf_val = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize([int(224 * scale_factor), int(224 * scale_factor)]),
            torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.7773, 0.7435, 0.7320],
                                             std=[0.2623, 0.2786, 0.2862])])
    return tsf_val(original_tensor)


class base_net(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_step_outputs = []
        self.validation_label = []
        self.train_step_outputs = []
        self.train_label = []

    def init_train_step(self, rate):
        def training_step(batch, batch_idx):
            if batch_idx == 0:
                self.train_step_outputs = []
                self.train_label = []
                gc.collect()
            img, label = batch
            embd = self(img)
            loss = self.criterion(embd, label)
            self.log("train_loss", loss)

            if random.random() < rate:
                self.train_step_outputs.extend(embd.cpu().detach().numpy())
                self.train_label.extend(label.tolist())
            return loss

        return training_step

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.validation_step_outputs = []
            self.validation_label = []
        img, label = batch
        embd = None
        # scales = [0.8, 1.0, 1.2]
        scales = self.scales
        for scale in scales:
            res = [sca_image(i, scale) for i in img]
            if embd is None:
                embd = self(torch.stack(res).to('cuda'))
            else:
                embd += self(torch.stack(res).to('cuda'))

        embd = embd / (len(scales) + 1)
        embd = l2_norm(embd.cpu().detach().numpy())

        loss = self.criterion(torch.from_numpy(embd).to('cuda'), label)
        self.validation_step_outputs.extend(embd)

        label = label.tolist()
        self.validation_label.extend(label)
        self.log("val_loss", loss)

    def init_validation_step(self, rate):
        def validation(batch, batch_idx):
            if batch_idx == 0:
                self.validation_step_outputs = []
                self.validation_label = []
                gc.collect()
            img, label = batch
            embd = self(img)
            loss = self.criterion(embd, label)
            self.log("val_loss", loss)
            if random.random() < rate:
                self.validation_label.extend(label.tolist())
                self.validation_step_outputs.extend(embd.cpu().detach().numpy())

        return validation

    def on_validation_epoch_end(self):
        self.log('seed', torch.tensor(float(os.environ["PL_GLOBAL_SEED"])))
        for tag, outputs, label, sample_count in [('val', self.validation_step_outputs, self.validation_label, 1000),
                                                  ('train', self.train_step_outputs, self.train_label, 1000)]:
            gc.collect()
            if not outputs:
                continue
            validation_step_outputs = torch.Tensor(np.array(outputs))
            sample = random.choices(validation_step_outputs, k=sample_count)
            sample = torch.stack(sample)
            # sample = validation_step_outputs
            # 传统方法
            rank, distance = self.my_neighbors(sample, validation_step_outputs)

            # mAp
            for i in [5, 10, 20, 30, 40, 50]:
                mAp = self.calc_map_neighbors_idxs(rank, top_k=i, validation_label=label)
                self.log(f"{tag}_mAp_{i}", mAp)

            # acc
            for i in [5, 10, 20, 30, 40, 50]:
                acc = self.calc_accuracy_neighbors_idxs(rank, top_k=i, validation_label=label)
                self.log(f"{tag}_acc_{i}", acc)

            # 聚类方法
            sample_label, cluster_data_idx = self.cluster(sample, outputs)
            neighbors_idxs = []
            for i, v in enumerate(sample_label):
                ranks = np.argsort(distance[i, cluster_data_idx[v]])
                neighbors_idxs.append([cluster_data_idx[v][j] for j in ranks])
            # acc
            for i in [5, 10, 20, 30, 40, 50]:
                acc = self.calc_accuracy_neighbors_idxs(neighbors_idxs, i, validation_label=label)
                self.log(f"{tag}_c_acc_{i}", acc)

            # map
            for i in [5, 10, 20, 30, 40, 50]:
                acc = self.calc_map_neighbors_idxs(neighbors_idxs, i, validation_label=label)
                self.log(f"{tag}_c_mAp_{i}", acc)

    def cluster(self, sample, datas):
        kmeans = KMeans(n_clusters=self.classes_num)
        # kmeans = MiniBatchKMeans(n_clusters=self.classes_num)
        kmeans.fit(datas)
        labels = kmeans.labels_
        cluster_data_idx = defaultdict(list)

        for i, v in enumerate(labels):
            cluster_data_idx[v].append(i)

        sample_labels = kmeans.predict(sample)
        return sample_labels, cluster_data_idx

    def normalize(self, line_vec):
        row_norms = np.linalg.norm(line_vec, axis=1)
        line_vec = line_vec / row_norms[:, np.newaxis]
        return line_vec

    def my_neighbors(self, sample, features):
        d1 = self.consine_distance(sample, features)
        # d1 = self.normalize(d1)
        # _, d2 = self.nan_euclidean_neighbors(sample, features)
        # d2 = self.normalize(d2)
        # _, d3 = self.manhattan_neighbors(sample, features)
        # d3 = self.normalize(d3)

        # distances = 1 * d1 + 0 * d2 + 0 * d3
        distances = d1
        ranks = np.argsort(distances, axis=1)
        return ranks, distances

    def consine_distance(self, sample, features):
        distance = sklearn.metrics.pairwise.cosine_distances(sample, features)
        return distance

    def nan_euclidean_distance(self, sample, features):
        distance = sklearn.metrics.pairwise.nan_euclidean_distances(sample, features)
        return distance

    def manhattan_distance(self, sample, features):
        distance = sklearn.metrics.pairwise.manhattan_distances(sample, features)
        return distance

    def euclidean_distance(self, sample, features):
        distance = sklearn.metrics.pairwise.euclidean_distances(sample, features)
        return distance

    def calc_accuracy_neighbors_idxs(self, neighbors_idxs, top_k, validation_label):
        macc = 0
        for neighbors_idx in neighbors_idxs:
            neighbors_idx = neighbors_idx[:top_k]
            if top_k in [5]:
                print(f'top-{top_k},{str(neighbors_idx)}')
            lbl = validation_label[neighbors_idx[0]]
            lbls = [validation_label[x] for x in neighbors_idx]
            macc += lbls.count(lbl) / top_k
        return macc / len(neighbors_idxs)

    def calc_accuracy_knn(self, sample, data, top_k):
        knn = NearestNeighbors(n_neighbors=top_k, metric='cosine')
        knn.fit(data)
        neighbors_idxs = knn.kneighbors(sample, return_distance=False)
        return self.calc_accuracy_neighbors_idxs(neighbors_idxs, top_k)

    def calc_map_neighbors_idxs(self, neighbors_idxs, top_k, validation_label):
        mAP = 0
        for neighbors_idx in neighbors_idxs:
            AP = 0
            lbl = validation_label[neighbors_idx[0]]
            now = 1
            for k, v in enumerate(neighbors_idx, start=1):
                # k:第几个 v:neighbors_idx nei的下标
                if now == top_k + 1:
                    break
                if validation_label[v] == lbl:
                    AP += now / k
                    now += 1
            mAP += AP / top_k
        return mAP / len(neighbors_idxs)

    def calc_mAP_knn(self, sample, data, top_k):
        knn = NearestNeighbors(n_neighbors=len(data), metric="cosine")
        knn.fit(data)
        neighbors_idxs = knn.kneighbors(sample, return_distance=False)
        return self.calc_map_neighbors_idxs(neighbors_idxs, top_k)
