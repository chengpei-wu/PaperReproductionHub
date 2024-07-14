import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=512, img_shape=(3, 64, 64)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.z_dim = z_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(z_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

    def sample(self, device="cuda"):
        z = torch.randn(1, self.z_dim).to(device)
        x = self.model(z)
        img = torch.reshape(x, (-1, *self.img_shape))
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity



if __name__ == '__main__':
    g = Generator()
    z = torch.rand(2, 512)
    print(z.shape)
    img = g(z)
    print(img.shape)
