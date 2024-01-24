import dgl
import torch.nn
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class GraphUnet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, ks):
        super().__init__()
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.ks = ks
        self.gpools = nn.ModuleList(
            [
                gPool(hidden_feats),
                gPool(hidden_feats),
            ]
        )
        self.gunpools = nn.ModuleList([gUnpool(), gUnpool()])
        self.gcn_layers = nn.ModuleList()
        for i in range(4):
            if i == 0:
                self.gcn_layers.append(
                    GraphConv(
                        in_feats,
                        hidden_feats,
                        activation=F.relu,
                        allow_zero_in_degree=True,
                    )
                )
            else:
                self.gcn_layers.append(
                    GraphConv(
                        hidden_feats,
                        hidden_feats,
                        activation=F.relu,
                        allow_zero_in_degree=True,
                    )
                )
        self.gcn_layers.append(
            GraphConv(
                hidden_feats,
                out_feats,
                activation=F.relu,
                allow_zero_in_degree=True,
            )
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        ori_graphs = []
        hidden_reps = []
        gpool_nids = []

        # down sampling
        for i in range(2):
            ori_graphs.insert(0, g.clone())
            h = self.gcn_layers[i](g, h)
            hidden_reps.insert(0, h)
            g, h, nids = self.gpools[i](g, h, (self.ks[i] * h.shape[0]))
            gpool_nids.insert(0, nids)

        h = self.gcn_layers[2](g, h)

        # up sampling
        for i in range(2):
            g, h = self.gunpools[i](ori_graphs[i], h, hidden_reps[i], gpool_nids[i])
            h = h + hidden_reps[i]
            h = self.gcn_layers[i + 3](g, h)

        return h


class gPool(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.projection = nn.Linear(in_feats, 1)

    def forward(self, g, h, top_k):
        scores = F.sigmoid(self.projection(h))
        g.ndata["scores"] = scores
        _, node_ids = dgl.topk_nodes(g, "scores", int(top_k), sortby=0)
        node_ids = torch.squeeze(node_ids)
        rst_g = dgl.node_subgraph(g, node_ids.int())
        new_h = h[node_ids]
        new_h = new_h * rst_g.ndata["scores"]
        return rst_g, new_h, node_ids


class gUnpool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ori_g, h, pre_h, selected_nids):
        new_h = pre_h.new_zeros((ori_g.number_of_nodes(), h.shape[1]))
        new_h[selected_nids] = h
        return ori_g, new_h


if __name__ == "__main__":
    g1 = dgl.graph(([0, 1], [2, 3]))
    g1.ndata["h"] = torch.rand(4, 1)
    print(g1.ndata["h"])
    _h, idx = dgl.topk_nodes(g1, "h", 3, sortby=0)
    g1 = dgl.node_subgraph(g1, torch.squeeze(idx), store_ids=True)
    print(g1.ndata["h"])
    g1 = dgl.add_nodes(g1, 2)
    print(g1.nodes())
