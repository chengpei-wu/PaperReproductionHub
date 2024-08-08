import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from GAN.model import Discriminator, Generator

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def train(device, dataloader, generator, discriminator):
    generator_optimizer = torch.optim.Adam(generator.parameters(), 5e-5)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), 1e-3)
    loss = nn.BCELoss()

    for epoch in range(200):
        discriminator_losses = []
        generator_losses = []

        for i, (x, _) in enumerate(dataloader):
            # train discriminator
            discriminator_optimizer.zero_grad()

            real_images = x.view(-1, 784).to(device)
            real_outputs = discriminator(real_images)

            z = torch.randn(x.shape[0], 128).to(device)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images)

            fake_labels = torch.zeros(x.shape[0], 1).to(device)
            real_labels = torch.ones(x.shape[0], 1).to(device)

            real_loss = loss(real_outputs, real_labels)
            fake_loss = loss(fake_outputs, fake_labels)

            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()
            discriminator_losses.append(discriminator_loss.item())

            # -------------------------------------------------
            # train generator
            generator_optimizer.zero_grad()

            z = torch.randn(x.shape[0], 128).to(device)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images)
            generator_loss = loss(fake_outputs, real_labels)
            generator_loss.backward()
            generator_optimizer.step()

            generator_losses.append(generator_loss.item())

        print(
            f'epoch-{epoch}: discriminator loss: {np.mean(discriminator_losses)}| generator loss: {np.mean(generator_losses)}')
        torch.save(generator.state_dict(), "demo_model.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 提取所有标签为9的样本
    indices = [i for i, (img, label) in enumerate(dataset) if label == 9]
    subset = Subset(dataset, indices)

    # 创建数据加载器
    dataloader = DataLoader(subset, batch_size=64, shuffle=False)

    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    generator, discriminator = Generator().to(device), Discriminator().to(device)
    train(device=device, dataloader=dataloader, generator=generator, discriminator=discriminator)
