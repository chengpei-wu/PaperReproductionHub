import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model import MLP


def load_dataset():
    data_train = pd.read_csv(f"./data/minst.csv")
    train_labels = data_train.loc[:, "label"].values
    train_imgs = data_train.iloc[:, 1:].values
    scaler = MinMaxScaler()
    train_imgs = scaler.fit_transform(train_imgs.T).T
    train_labels = list(map(int, train_labels.reshape((len(train_labels),)).tolist()))
    # 分类问题，将y设置为one-hot编码
    train_labels = np.eye(10)[train_labels]
    return np.array(train_imgs), np.array(train_labels)


if __name__ == "__main__":

    x, y = load_dataset()
    # 加载minst数据集

    print(x.shape, y.shape)
    mlp = MLP(in_feats=x.shape[1], hid_feats=64, out_feats=y.shape[1], num_layer=3)

    for epoch in range(10):
        for i in range(len(x)):
            index = random.randint(0, len(x) - 1)
            mlp.back_propagation(
                x[index],
                y[index],
                0.01,
            )
        accuracy, loss = mlp.score(x, y)
        print(f"Epoch:{epoch}, Accuracy: {accuracy:.3f}, Loss: {loss:.3f}.")
