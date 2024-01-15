import torch.nn as nn
import torch as th
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLError
from dgl.nn.pytorch import GraphConv
from dgl.utils import expand_as_pair


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
