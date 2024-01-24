import dgl.sparse
import networkx as nx
import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import random_split

from dataset import SyntheticDataset
from model import VGAE
import torch.nn.functional as F


def kl_divergence(z, mu, log_sigma):
    return (
        0.5
        / z.size(0)
        * (1 + 2 * log_sigma - mu**2 - torch.exp(log_sigma) ** 2).sum(1).mean()
    )


def train(
    device: torch.device,
    model: torch.nn.Module,
    dataset,
    max_epoch: int,
    batch_size: int,
):
    train_loader = GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    model.to(device)

    for epoch in range(max_epoch):
        model.train()
        total_loss = 0
        for batch, batched_graph in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            h = batched_graph.ndata.pop("deg_feat")
            logits = model(batched_graph, h, device)
            adj = batched_graph.adj().to_dense()
            loss = F.binary_cross_entropy(logits.view(-1), adj.view(-1))
            divergence = kl_divergence(logits, model.mu, model.log_sigma)
            loss = loss - divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch}--Loss: {total_loss / (batch + 1):.4f}")


if __name__ == "__main__":
    # dataset
    nets = SyntheticDataset(
        network_type="er", average_degree=(3, 10), num_nodes=100, num_graphs=10
    )
    train_size = int(0.9 * len(nets))
    val_size = len(nets) - train_size
    train_dataset, val_dataset = random_split(nets, [train_size, val_size])

    # train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = VGAE(
        in_feats=1,
        hidden_feats=16,
        out_feats=2,
    ).to(device)
    train(device=device, model=model, dataset=train_dataset, max_epoch=5, batch_size=1)
    torch.save(model, "./model")
