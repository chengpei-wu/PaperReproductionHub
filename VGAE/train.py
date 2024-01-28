# import argparse
#
# import dgl
# import dgl.function as fn
# import torch
# import torch.nn as nn
# from dgl import AddSelfLoop
# from dgl.data import CiteseerGraphDataset
# from dgl.data import CoraGraphDataset
# from dgl.data import PubmedGraphDataset
# from torch import optim
#
# from model import VGAE
# import torch.nn.functional as F
#
#
# class DotProductPredictor(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, graph, h):
#         with graph.local_scope():
#             graph.ndata["h"] = h
#             graph.apply_edges(fn.u_dot_v("h", "h", "score"))
#             return graph.edata["score"]
#
#
# def kl_divergence(z, mu, log_sigma):
#     return (
#         0.5
#         / z.size(0)
#         * (1 + 2 * log_sigma - mu**2 - torch.exp(log_sigma) ** 2).sum(1).mean()
#     )
#
#
# def train(
#     g,
#     features,
#     masks,
#     model,
#     max_epoch,
# ):
#     train_mask = masks[0]
#     val_mask = masks[1]
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
#     model.to(device)
#
#     for epoch in range(max_epoch):
#         model.train()
#         logits = model(g, features)
#         adj = g.adj().to_dense()
#         loss = F.binary_cross_entropy(logits.view(-1), adj.view(-1))
#         divergence = kl_divergence(logits, model.mu, model.log_sigma)
#         loss = loss - divergence
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         print(f"Epoch {epoch}--Loss: {loss.item():.4f}")
#
#
# # Implementation based on Dropping all edges during training GNN.
#
# if __name__ == "__main__":
#     args_list = [
#         ("--dataset", "pubmed"),
#         ("--datatype", "float"),
#         ("--hid_size", 16),
#         ("--lr", 1e-2),
#         ("--max_epoch", 200),
#     ]
#
#     # argument configuration
#     parser = argparse.ArgumentParser()
#
#     for arg_name, default in args_list:
#         parser.add_argument(arg_name, default=default)
#     args = parser.parse_args()
#
#     # load and preprocess dataset
#     if args.dataset == "cora":
#         if args.self_loop:
#             data = CoraGraphDataset(transform=AddSelfLoop())
#         else:
#             data = CoraGraphDataset()
#
#     elif args.dataset == "citeseer":
#         if args.self_loop:
#             data = CiteseerGraphDataset(transform=AddSelfLoop())
#         else:
#             data = CiteseerGraphDataset()
#     elif args.dataset == "pubmed":
#         if args.self_loop:
#             data = PubmedGraphDataset(transform=AddSelfLoop())
#         else:
#             data = PubmedGraphDataset()
#     else:
#         raise ValueError("Unknown dataset: {}".format(args.dataset))
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     g = data[0]
#     g = g.int().to(device)
#
#     features = g.ndata["feat"]
#     labels = g.ndata["label"]
#     masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
#
#     # create GCN model
#     in_size = features.shape[1]
#     out_size = data.num_classes
#
#     model = VGAE(in_size, args.hid_size, out_size).to(device)
#
#     # convert model and graph to bfloat16 if needed
#     if args.datatype == "bfloat16":
#         g = dgl.to_bfloat16(g)
#         features = features.to(dtype=torch.bfloat16)
#         model = model.to(dtype=torch.bfloat16)
#
#     # model training
#     train(g, features, labels, masks, model, lr=args.lr, max_epoch=args.max_epoch)
