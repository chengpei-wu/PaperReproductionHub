import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class ControlledGraphConv(GraphConv):
    def __init__(
        self,
        in_feats,
        out_feats,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super().__init__(
            in_feats,
            out_feats,
            norm,
            weight,
            bias,
            activation,
            allow_zero_in_degree,
        )

    def forward(
        self, graph, feat, weight=None, edge_weight=None, allow_message_passing=True
    ):
        if allow_message_passing:
            return super().forward(graph, feat, weight, edge_weight)
        else:
            if self.weight is not None:
                weight = self.weight
            else:
                raise ValueError(
                    "weight must be initialized when allowing message passing."
                )
            rst = th.matmul(feat, weight)

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class ControlledGCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    ControlledGraphConv(
                        in_size, hid_size, allow_zero_in_degree=True, activation=F.relu
                    )
                )
            elif i == num_layers - 1:
                self.layers.append(
                    ControlledGraphConv(hid_size, out_size, allow_zero_in_degree=True)
                )
            else:
                self.layers.append(
                    ControlledGraphConv(
                        hid_size, hid_size, allow_zero_in_degree=True, activation=F.relu
                    )
                )

        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features, allow_message_passing=True):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, allow_message_passing=allow_message_passing)
        return h


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(
                    GraphConv(
                        in_size, hid_size, allow_zero_in_degree=True, activation=F.relu
                    )
                )
            elif i == num_layers - 1:
                self.layers.append(
                    GraphConv(hid_size, out_size, allow_zero_in_degree=True)
                )
            else:
                self.layers.append(
                    GraphConv(
                        hid_size, hid_size, allow_zero_in_degree=True, activation=F.relu
                    )
                )

        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
