import torch
import torch.nn as nn
from tqdm import tqdm

from GAN.data_loader import get_dataloader
from GAN.model import Discriminator, Generator

# Hyperparameters
n_epochs = 100
lr = 0.01
img_shape=(3, 64, 64)
z_dim = 512


def train(device, dataloader, generator, discriminator):
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(n_epochs)):

        discriminator_losses = []
        correct_predictions = 0
        total_predictions = 0

        # train generator
        for x in dataloader:
            real_labels = torch.ones(x.shape[0], 1).to(device)
            random_guassian_noise = torch.randn(x.shape[0], z_dim).to(device)
            fake_images = generator(random_guassian_noise)
            fake_outputs = discriminator(fake_images)

            generator_loss = criterion(fake_outputs, real_labels)

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            fake_labels = torch.zeros(x.shape[0], 1).to(device)
            labels = fake_labels.squeeze()
            predictions = torch.round(fake_outputs.squeeze())
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions

        print(f"Epoch {epoch + 1}/{n_epochs}, Accuracy on fake images: {accuracy:.4f}")

        # train discriminator
        if accuracy < 0.5:
            for x in dataloader:
                real_images = x.to(device)
                random_guassian_noise = torch.randn(x.shape[0], z_dim).to(device)
                fake_images = generator(random_guassian_noise)

                real_labels = torch.ones(x.shape[0], 1).to(device)
                fake_labels = torch.zeros(x.shape[0], 1).to(device)

                real_outputs = discriminator(real_images)
                fake_outputs = discriminator(fake_images)

                real_loss = criterion(real_outputs, real_labels)
                fake_loss = criterion(fake_outputs, fake_labels)

                discriminator_loss = real_loss + fake_loss
                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()

        torch.save(generator.state_dict(), "demo_model.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = get_dataloader()
    generator, discriminator = Generator(z_dim=z_dim, img_shape=img_shape).to(
        device), Discriminator(img_shape=img_shape).to(device)
    train(device=device, dataloader=data_loader, generator=generator, discriminator=discriminator)
