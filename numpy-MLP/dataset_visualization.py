import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    data_train = pd.read_csv(f"./data/minst.csv")
    train_labels = data_train.loc[:, "label"].values
    train_imgs = data_train.iloc[:, 1:].values.reshape(-1, 28, 28)

    images = [train_imgs[i] for i in range(36)]

    merged_image = np.vstack([np.hstack(images[i : i + 6]) for i in range(0, 36, 6)])

    # 显示合并后的图像
    plt.imshow(merged_image, cmap="gray")
    plt.axis("off")  # 关闭坐标轴
    plt.show()
