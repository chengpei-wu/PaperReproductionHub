import argparse

import dgl
import torch
import torch.nn as nn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from model import ControlledGCN, GCN


def evaluate(g, features, labels, mask, model, allow_message_passing):
    model.eval()
    with torch.no_grad():
        logits = model(g, features, allow_message_passing)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(
    g,
    features,
    labels,
    masks,
    model,
    mlp_lr,
    lr,
    max_epoch,
    use_massage_passing,
    self_loop,
):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=mlp_lr, weight_decay=5e-4)

    # training loop

    allow_message_passing = False
    if self_loop:
        g = dgl.add_self_loop(g)
    for epoch in range(max_epoch):
        model.train()
        if epoch >= int(max_epoch * use_massage_passing) and not allow_message_passing:
            allow_message_passing = True
            optimizer.lr = lr

        logits = model(g, features, allow_message_passing)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = evaluate(
            g, features, labels, train_mask, model, allow_message_passing
        )
        val_acc = evaluate(g, features, labels, val_mask, model, allow_message_passing)
        print(
            "\r",
            f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc. {train_acc:.4f} | Val. Acc. {val_acc:.4f}",
            flush=True,
            end="",
        )


# Implementation based on defining new GNN layer, which allows to use weight matrix only.

if __name__ == "__main__":
    args_list = [
        ("--dataset", "cora"),
        (
            "--self_loop",
            True,
        ),  # must add self loop, if not, the feats will be zeros (when drop all edges) during training.
        ("--datatype", "float"),
        ("--num_layers", 2),
        ("--hid_size", 16),
        ("--mlp_lr", 1e-2),
        ("--gnn_lr", 1e-2),
        ("--max_epoch", 200),
        (
            "--when_message_passing",
            1,
        ),
        # '1' means training MLP only
        # '0.8' means using message passing when epoch == max_epoch*0.8
        # '0' means using message passing during training
    ]

    # argument configuration
    parser = argparse.ArgumentParser()

    for arg_name, default in args_list:
        parser.add_argument(arg_name, default=default)
    args = parser.parse_args()

    # load and preprocess dataset
    if args.dataset == "cora":
        if args.self_loop:
            data = CoraGraphDataset(transform=AddSelfLoop())
        else:
            data = CoraGraphDataset()

    elif args.dataset == "citeseer":
        if args.self_loop:
            data = CiteseerGraphDataset(transform=AddSelfLoop())
        else:
            data = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        if args.self_loop:
            data = PubmedGraphDataset(transform=AddSelfLoop())
        else:
            data = PubmedGraphDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = data[0]
    g = g.int().to(device)

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # create GCN model
    in_size = features.shape[1]
    out_size = data.num_classes

    model = ControlledGCN(in_size, args.hid_size, out_size, args.num_layers).to(device)

    # convert model and graph to bfloat16 if needed
    if args.datatype == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # model training
    train(
        g,
        features,
        labels,
        masks,
        model,
        mlp_lr=args.mlp_lr,
        lr=args.gnn_lr,
        max_epoch=args.max_epoch,
        use_massage_passing=args.when_message_passing,
        self_loop=args.self_loop,
    )

    # test the model without message passing
    acc1 = evaluate(g, features, labels, masks[2], model, False)

    # test the model with message passing
    acc2 = evaluate(g, features, labels, masks[2], model, True)
    print("\n\n")
    if 0 < args.when_message_passing < 1:
        print(
            f"Training MLP for GNN initial training: {int(args.max_epoch * args.when_message_passing)} epochs for MLP, the last {int(args.max_epoch * (1 - args.when_message_passing))} epochs adding message passing."
        )
    elif args.when_message_passing == 0:
        print(f"Training GNN only.")
    else:
        print(f"Training MLP only.")

    print(
        f"Testing accuracy for {args.dataset}: \n\t with message passing: {acc2}\n\t without message passing: {acc1}"
    )
