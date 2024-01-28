import argparse

import dgl
import torch
import torch.nn as nn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset
from dgl.data import CoraGraphDataset
from dgl.data import PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

from model import GraphUnet


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model, lr, max_epoch):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss(label_smoothing=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # training loop
    losses, accuracies = [], []
    for epoch in range(max_epoch):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = evaluate(
            g,
            features,
            labels,
            train_mask,
            model,
        )
        val_acc = evaluate(
            g,
            features,
            labels,
            val_mask,
            model,
        )
        print(
            "\r",
            f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc. {train_acc:.4f} | Val. Acc. {val_acc:.4f}",
            flush=True,
            end="",
        )


if __name__ == "__main__":
    # argument configuration
    parser = argparse.ArgumentParser()
    args_list = [
        ("--dataset", "cora"),
        ("--datatype", "float"),
        ("--hid_size", 512),
        ("--embed_size", 48),
        ("--lr", 1e-2),
        ("--max_epoch", 200),
    ]

    for arg_name, default in args_list:
        parser.add_argument(arg_name, default=default)
    args = parser.parse_args()

    # load and preprocess dataset
    transform = AddSelfLoop()
    is_ogb = False
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    elif args.dataset == "ogbn-arxiv":
        data = DglNodePropPredDataset(name="ogbn-arxiv")
        is_ogb = True
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not is_ogb:
        g = data[0]
        g = g.int().to(device)
        features = g.ndata["feat"]
        labels = g.ndata["label"]
        masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    else:
        g, labels = data[0]
        g = dgl.add_self_loop(g)
        g = g.int().to(device)
        labels = labels.squeeze().to(device)
        features = g.ndata["feat"]
        split_idx = data.get_idx_split()
        masks = split_idx["train"], split_idx["valid"], split_idx["test"]
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GraphUnet(
        in_feats=in_size,
        embed_feats=args.embed_size,
        hidden_feats=args.hid_size,
        out_feats=out_size,
        ks=[2000, 1000, 500],
    ).to(device)

    # convert model and graph to bfloat16 if needed
    if args.datatype == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # model training
    train(g, features, labels, masks, model, lr=args.lr, max_epoch=args.max_epoch)

    # test the model
    acc = evaluate(g, features, labels, masks[2], model)
    print("\nTest accuracy {:.4f}".format(acc))
