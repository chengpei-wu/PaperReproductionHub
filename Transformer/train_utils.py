import time

import torch


def subsequent_mask(size):
    # attention mask for unseen words (tokens)
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class Batch:
    # construct batch for training
    def __init__(self, src, tgt=None, pad=2, device=torch.device("cpu")):  # 2 = <blank>
        self.device = device

        # input for encoder
        self.src = src.to(device)

        # add padding mask for sentences that shorter than max_len
        self.src_mask = (src != pad).unsqueeze(-2).to(device)

        # add padding mask and future unseen
        if tgt is not None:
            # initial input for decoder
            self.tgt = tgt[:, :-1].to(device)

            # expected output for transformer (decoder)
            self.tgt_y = tgt[:, 1:].to(device)

            # add masks for decoder input (both padding mask and future mask)
            self.tgt_mask = self.make_std_mask(self.tgt, pad)

            # the number of tokens in a batch
            self.ntokens = (self.tgt_y != pad).data.sum().to(device)

    def make_std_mask(self, tgt, pad):
        # Create a mask to hide padding and future words
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask.to(self.device)


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    # Train a single epoch
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for epoch, batch in enumerate(data_iter):
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if epoch % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if epoch % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                f"Epoch Step: {epoch} | Accumulation Step: {n_accum} | Loss: {loss / batch.ntokens:6.2f} "
                + f"| Tokens / Sec: {tokens / elapsed:7.1f} | Learning Rate: {lr:6.1e}"
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data).to(device)
    for i in range(max_len - 1):
        out = model.decode(
            ys,
            memory,
            src_mask,
            subsequent_mask(ys.size(1)).type_as(src.data).to(device),
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


if __name__ == "__main__":
    print(subsequent_mask(10))
