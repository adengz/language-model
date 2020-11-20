import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score

from utils import timeit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss_fn(neg_embed, hidden, pos_embed):
    """

    Args:
        neg_embed: seq_len * batch_size, sample_size, embedding_dim
        hidden: seq_len * batch_size, embedding_dim, 1
        pos_embed: seq_len * batch_size, 1, embedding_dim

    Returns:
        0-dim loss
    """
    pos_score = torch.bmm(pos_embed, hidden).squeeze()  # seq_len * batch_size
    pos_contrib = -F.logsigmoid(pos_score).mean()

    neg_score = torch.bmm(neg_embed, hidden).squeeze()  # seq_len * batch_size, sample_size
    neg_contrib = -torch.log(1 - torch.sigmoid(neg_score)).mean()

    return pos_contrib + neg_contrib


def get_predictions(vocab_embed, hidden):
    """

    Args:
        hidden: seq_len * batch_size, embedding_dim, 1
        vocab_embed: 1, vocab_size, embedding_dim

    Returns:
        (seq_len * batch_size)
    """
    count = hidden.shape[0]
    vocab_embed, _ = torch.broadcast_tensors(vocab_embed, torch.Tensor(count, 1, 1))
    # seq_len * batch_size, vocab_size, embedding_dim
    score = torch.bmm(vocab_embed, hidden).squeeze()  # seq_len * batch_size, vocab_size
    return score.argmax(1)


@timeit
@torch.no_grad()
def evaluate(model, dataloader, pad_idx, vocab_tensor):
    model.eval()

    epoch_loss = epoch_acc = total_count = non_pad_count = 0
    for inputs, targets, _, neg_samples in dataloader:
        inputs, targets, neg_samples = inputs.to(device), targets.to(device), neg_samples.to(device)

        neg_embed, hidden, pos_embed = model(neg_samples, inputs, targets)
        loss = loss_fn(neg_embed, hidden, pos_embed)
        vocab_embed = model(vocab_tensor)[0]
        targets = targets.view(-1)
        y = np.vstack([get_predictions(vocab_embed, hidden).cpu().numpy(), targets.cpu().numpy()])
        # 2, seq_len * batch_size
        y = y[:, np.where(y[1, :] != pad_idx)[0]]  # 2, filtered_non_padding
        acc = accuracy_score(y[0], y[1])

        total_count += len(targets)  # seq_len * batch_size
        epoch_loss += loss.item() * len(targets)
        non_pad_count += len(y[1])  # filtered_non_padding
        epoch_acc += acc * len(y[1])

    model.train()
    return epoch_loss / total_count, epoch_acc / non_pad_count


@timeit
def train_1_epoch(model, dataloader, optimizer, pad_idx, vocab_tensor):
    model.train()

    epoch_loss = epoch_acc = total_count = non_pad_count = 0
    for inputs, targets, _, neg_samples in dataloader:
        inputs, targets, neg_samples = inputs.to(device), targets.to(device), neg_samples.to(device)
        optimizer.zero_grad()

        neg_embed, hidden, pos_embed = model(neg_samples, inputs, targets)
        loss = loss_fn(neg_embed, hidden, pos_embed)
        vocab_embed = model(vocab_tensor)[0]
        targets = targets.view(-1)
        y = np.vstack([get_predictions(vocab_embed, hidden).cpu().numpy(), targets.cpu().numpy()])
        # 2, seq_len * batch_size
        y = y[:, np.where(y[1, :] != pad_idx)[0]]  # 2, filtered_non_padding
        acc = accuracy_score(y[0], y[1])

        loss.backward()
        optimizer.step()

        total_count += len(targets)  # seq_len * batch_size
        epoch_loss += loss.item() * len(targets)
        non_pad_count += len(y[1])  # filtered_non_padding
        epoch_acc += acc * len(y[1])

    return epoch_loss / total_count, epoch_acc / non_pad_count


def train_model(model, filename, train_loader, val_loader, vocab_size, pad_idx, optim, lr=1e-3, epochs=20):
    model.to(device)
    optimizer = optim(model.parameters(), lr=lr)
    vocab_tensor = torch.arange(vocab_size).to(device)
    min_val_loss = float('inf')

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1: 02}')

        train_loss, train_acc = train_1_epoch(model, train_loader, optimizer, pad_idx, vocab_tensor)
        print(f'\tTrain Loss: {train_loss: .3f} | Train Acc: {train_acc * 100: .2f}%')

        val_loss, val_acc = evaluate(model, val_loader, pad_idx, vocab_tensor)
        print(f'\t Val. Loss: {val_loss: .3f} |  Val. Acc: {val_acc * 100: .2f}%')

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'{filename}')
            print(f'\tModel parameters saved to {filename}')
