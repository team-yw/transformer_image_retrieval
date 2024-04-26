import random

import numpy as np


class Kmedoid:
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def randCent(self):  # 随机选取一个点
        random_index = random.randint(0, self.data.shape[0] - 1)
        return random_index, self.data[random_index, :]

    def distance(self, vecA, vecB):  # 计算曼哈顿距离
        return sum(abs(vecA - vecB))

    def run(self):
        init_centers = []  # 初始化中心的列表
        init_indexs = []  # 被选中作为中心的点的下标
        while len(init_centers) < self.k:
            index, center = self.randCent()
            if index not in init_indexs:  # 保证选点不重复
                init_centers.append(center)
                init_indexs.append(index)
            else:
                continue

        while True:
            cluster_category = []  # 记录聚类结果
            for i in range(self.data.shape[0]):  # 遍历每一个点
                minv = np.inf  # 最小距离，初始为正无穷
                cluster_index = 0  # 所属簇的下标
                for index, center in enumerate(init_centers):  # 遍历每个中心
                    # 选取离得最近的中心作为归属簇
                    dist = self.distance(center, self.data[i, :])
                    if dist < minv:
                        minv = dist
                        cluster_index = index
                cluster_category.append(cluster_index)

            # 重新计算中心点
            new_indexs = [0 for i in range(len(init_centers))]  # 更新被选中作为中心的点的下标
            min_dists = [np.inf for i in range(len(init_centers))]  # 中心点对应最小距离
            for i in range(self.data.shape[0]):
                min_dist = 0  # 求与当前簇其他点的距离之和
                for j in range(self.data.shape[0]):  # 遍历每一个点
                    if cluster_category[i] == cluster_category[j]:  # 属于同一个簇才进行累加
                        min_dist += self.distance(self.data[i, :], self.data[j, :])
                if min_dist < min_dists[cluster_category[i]]:  # 保存数据到列表
                    min_dists[cluster_category[i]] = min_dist
                    new_indexs[cluster_category[i]] = i

            init_centers = []  # 新的聚类中心
            for index in new_indexs:
                init_centers.append(self.data[index, :])

            if new_indexs == init_indexs:  # 如果新的中心与上次相同则结束循环
                return cluster_category, init_centers
            else:
                init_indexs = new_indexs  # 更新聚类中心下标
