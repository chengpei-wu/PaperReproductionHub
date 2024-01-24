import dgl
import networkx as nx
import numpy as np
from dgl.data import DGLDataset
from tqdm import tqdm


class SyntheticDataset(DGLDataset):
    def __init__(self, network_type, average_degree: tuple, num_nodes, num_graphs=1000):
        self.graphs = None
        self.network_type = network_type
        self.average_degree = average_degree
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        super().__init__(name=f"SyntheticDataset_{network_type}")

    def process(self):
        graphs = []
        for _ in tqdm(
            range(self.num_graphs), desc=f"generating {self.network_type} nets"
        ):
            if self.network_type == "er":
                num_edges = int(
                    np.random.uniform(*self.average_degree) * self.num_nodes * 0.5
                )
                nx_g = nx.gnm_random_graph(self.num_nodes, num_edges)
            else:
                raise NotImplementedError(
                    f"network type {self.network_type} not implemented yet."
                )
            _g = dgl.add_self_loop(dgl.from_networkx(nx_g))
            _g.ndata["deg_feat"] = _g.in_degrees().view(-1, 1).float()
            graphs.append(_g)

        self.graphs = graphs

    def __getitem__(self, idx) -> dgl.DGLGraph:
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
