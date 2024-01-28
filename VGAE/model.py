import dgl.nn.pytorch as dglnn
import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class VGAE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(VGAE, self).__init__()
        self.log_sigma = None
        self.mu = None
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        graph_conv_layers1 = dglnn.GraphConv(
            in_feats, hidden_feats, activation=F.relu, allow_zero_in_degree=True
        )
        graph_conv_layers2 = dglnn.GraphConv(
            hidden_feats, out_feats, allow_zero_in_degree=True
        )
        graph_conv_layers3 = dglnn.GraphConv(
            hidden_feats, out_feats, allow_zero_in_degree=True
        )
        self.gcn = nn.ModuleList(
            [graph_conv_layers1, graph_conv_layers2, graph_conv_layers3]
        )

    def encoder(self, g, feats, device):
        h = feats
        h = self.gcn[0](g, h)
        self.mu = self.gcn[1](g, h)
        self.log_sigma = self.gcn[2](g, h)

        # reparameterize
        std_norm_noise = torch.randn(size=(feats.shape[0], self.out_feats)).to(device)
        sampled_latent_feats = self.mu + (
            std_norm_noise * torch.exp(self.log_sigma).to(device)
        )
        # N(mu, sigma^2) ~ mu + N(0, 1)*sigma
        return sampled_latent_feats

    def decoder(self, z):
        return torch.sigmoid(z @ z.T)

    def forward(self, g, feats, device):
        latent_feats = self.encoder(g, feats, device)
        adj = self.decoder(latent_feats)
        return adj


if __name__ == "__main__":
    vgae = VGAE(1, 16, 2)
    print(vgae.gcn)
