import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from sklearn.datasets._samples_generator import make_blobs


def generate_samples(n_samples, centers):
    return make_blobs(n_samples=n_samples, centers=centers, random_state=1234)


def show_samples(x, y, clusters_centroids):
    plt.title("samples")
    plt.scatter(x[:, 0], x[:, 1], color=[f"C{label}" for label in y])
    plt.scatter(
        clusters_centroids[:, 0], clusters_centroids[:, 1], c="black", marker="+"
    )
    plt.xlabel("feature dimension 1")
    plt.ylabel("feature dimension 2")
    plt.show()


def show_animation(x, y, history_clusters_centroids):
    fig = plt.figure()
    plt.scatter(x[:, 0], x[:, 1], color=[f"C{label}" for label in y])
    point_ani = plt.scatter(
        history_clusters_centroids[0, :, 0],
        history_clusters_centroids[0, :, 1],
        c="black",
        marker="+",
    )

    def update_points(i):
        point_ani.set_offsets(history_clusters_centroids[i, :, :])
        return (point_ani,)

    ani = animation.FuncAnimation(
        fig, update_points, np.arange(0, len(history_clusters_centroids)), interval=500
    )
    plt.show()
