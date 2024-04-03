import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from Transformer.model import Transformer
from Transformer.train_utils import Batch
from Transformer.train_utils import greedy_decode
from Transformer.train_utils import run_epoch
from Transformer.train_utils import subsequent_mask


def data_gen(V, batch_size, nbatches, device):
    # Generate random data for a src-tgt copy task.
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0, device=device)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, x, y, norm):
        sloss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        return sloss.data * norm, sloss


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


if __name__ == "__main__":
    # Train the simple copy task for transformer.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def example_simple_model():
        V = 11
        criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        model = Transformer(V, V, num_layer=2).to(device=device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, model_size=model.src_embeding.d_model, factor=1.0, warmup=400
            ),
        )

        batch_size = 256
        for epoch in range(30):
            model.train()
            run_epoch(
                data_gen(V, batch_size, 20, device=device),
                model,
                SimpleLossCompute(criterion),
                optimizer,
                lr_scheduler,
                mode="train",
            )
            model.eval()
            t = run_epoch(
                data_gen(V, batch_size, 5, device=device),
                model,
                SimpleLossCompute(criterion),
                DummyOptimizer(),
                DummyScheduler(),
                mode="eval",
            )[0]

        model.eval()
        src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(device)
        max_len = src.shape[1]
        src_mask = torch.ones(1, 1, max_len).to(device)
        print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))

    example_simple_model()
