import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat


def findClosestCentroids(X, centroids):
    c = np.empty((0, X.shape[0]))
    for i in range(len(centroids)):
        c = np.vstack((c, np.sum((X - centroids[i])**2, axis=1)))
    idx = np.argmin(c, axis=0)
    return idx


def computeCentroids(X, centroids, idx):
    new_centroids = np.empty((0, centroids.shape[1]))
    for i in range(len(centroids)):
        c = X[idx == i]
        new_centroids = np.vstack((new_centroids, c.mean(axis=0)))
    return new_centroids


def k_means(X, K, epoch):
    randidx = np.random.randint(1, X.shape[0])
    centroids = X[randidx - K:randidx]  #随机初始化，取随机索引的前K个样本作为初始化聚类中心
    for i in range(epoch):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, centroids, idx)
    return centroids, idx


def show_k_means(input, K=3, epoch=500):
    centorids, idx = k_means(input, K, epoch)
    clusters = []
    for i in range(K):
        clusters.append("cluster{}".format(i))
    fig, axarr = plt.subplots(1, 2, figsize=(21, 8))
    for i in range(K):
        clusters[i] = X[np.where(idx == i)[0], :]
        axarr[1].scatter(clusters[i][:, 0],
                         clusters[i][:, 1],
                         s=30,
                         cmap='rainbow',
                         label='Cluster{}'.format(i))
    axarr[1].legend()
    axarr[0].scatter(X[:, 0], X[:, 1])
    plt.show()


data = loadmat("D:\study\code\ML\ex7_k-means\data\ex7data2.mat")
X = data["X"]
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = findClosestCentroids(X, initial_centroids)

show_k_means(X, K=3)
