from scipy.spatial.distance import cdist

from utils import *


class K_Means:
    def __init__(self, n_clusters=4, max_iter=1000, centroids=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.history_centroids = []
        if not centroids:
            self.centroids = np.random.rand(n_clusters, 2)
        else:
            self.centroids = centroids

    def fit(self, x):
        for iter in range(self.max_iter):
            self.history_centroids.append(self.centroids.tolist())
            print("\r", f"{iter + 1} / {self.max_iter}", end="", flush=True)
            clusters = dict()
            distance = cdist(x, self.centroids)
            classes = np.argmin(distance, axis=1)
            for point, cls in zip(x, classes):
                if not clusters.__contains__(cls):
                    clusters[cls] = []
                clusters[cls].append(point.tolist())
            for cls, points in zip(clusters.keys(), clusters.values()):
                points = np.array(points)
                self.centroids[cls] = np.array(
                    [np.mean(points[:, 0]), np.mean(points[:, 1])]
                )

    def predict(self, x):
        distance = cdist(x, self.centroids)
        cls = np.argmin(distance, axis=1)
        return cls

    def get_centroids(self):
        return self.centroids

    def get_history_centroids(self):
        return np.array(self.history_centroids)
