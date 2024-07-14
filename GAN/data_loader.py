import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class CelebADataset(Dataset):
    def __init__(self, root, img_shape=(64, 64)) -> None:
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))
        self.memory_imgs = []
        self.pipeline = transforms.Compose(
            [
                transforms.CenterCrop(168),
                transforms.Resize(self.img_shape),
                transforms.ToTensor(),
            ]
        )
        self.init_memory_images()

    def __len__(self) -> int:
        return len(self.filenames) // 10

    def init_memory_images(self):
        for i in tqdm(range(len(self.filenames) // 10), desc="loading dataset into memory"):
            path = os.path.join(self.root, self.filenames[i])
            self.memory_imgs.append(self.pipeline(Image.open(path).convert("RGB")))

    def __getitem__(self, index: int):
        return self.memory_imgs[index]


def get_dataloader(root="D:/WorkSpace/dataset/img_align_celeba"):
    dataset = CelebADataset(root=root)
    return DataLoader(dataset, 16, shuffle=True, pin_memory=True)


if __name__ == "__main__":
    dataloader = get_dataloader()
    img = next(iter(dataloader))
    print(img.shape)
    N, C, H, W = img.shape
    assert N == 16
    img = torch.permute(img, (1, 0, 2, 3))
    img = torch.reshape(img, (C, 4, 4 * H, W))
    img = torch.permute(img, (0, 2, 1, 3))
    img = torch.reshape(img, (C, 4 * H, 4 * W))
    img = transforms.ToPILImage()(img)
    img.save("tmp.jpg")
