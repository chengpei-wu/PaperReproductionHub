from time import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from VAE.model import VAE
from data_loader import get_dataloader

# Hyperparameters
n_epochs = 10
kl_weight = 0.00025
lr = 0.005


def loss_fn(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0
    )
    loss = recons_loss + kl_loss * kl_weight
    return loss


def train(device, dataloader, model):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset_len = len(dataloader.dataset)
    begin_time = time()
    for i in range(n_epochs):
        loss_sum = 0
        for x in dataloader:
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        loss_sum /= dataset_len
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f"epoch {i}: loss {loss_sum} {minute}:{second}")
        torch.save(model.state_dict(), "demo_model.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = get_dataloader()
    model = VAE()
    train(device=device, dataloader=data_loader, model=model)
