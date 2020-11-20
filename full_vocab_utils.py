from collections import Counter

import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

from utils import timeit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss().to(device)


@timeit
@torch.no_grad()
def evaluate(model, dataloader, loss_func, pad_idx, read_prev=False):
    model.eval()

    epoch_loss = epoch_acc = total_count = non_pad_count = 0
    wrong_preds = Counter()
    for inputs, targets, prevs, _ in dataloader:
        inputs, targets, prevs = inputs.to(device), targets.to(device), prevs.to(device)

        hidden = model(prevs)[1] if read_prev else None
        outputs, _ = model(inputs, hidden)
        targets = targets.view(-1)

        loss = loss_func(outputs, targets)
        y = np.vstack([outputs.argmax(1).cpu().numpy(), targets.cpu().numpy()])  # 2, seq_len * batch_size
        y = y[:, np.where(y[1, :] != pad_idx)[0]]  # 2, filtered_non_padding
        acc = accuracy_score(y[0], y[1])
        mismatches = y[:, np.where(y[0, :] != y[1, :])[0]]  # 2, filtered_wrong_prediction
        wrong_preds += Counter(map(tuple, mismatches.T))

        total_count += len(targets)  # seq_len * batch_size
        epoch_loss += loss.item() * len(targets)
        non_pad_count += len(y[1])  # filtered_non_padding
        epoch_acc += acc * len(y[1])

    model.train()
    return epoch_loss / total_count, epoch_acc / non_pad_count, wrong_preds


@timeit
def train_1_epoch(model, dataloader, loss_func, optimizer, pad_idx, read_prev=False):
    model.train()

    epoch_loss = epoch_acc = total_count = non_pad_count = 0
    for inputs, targets, prevs, _ in dataloader:
        inputs, targets, prevs = inputs.to(device), targets.to(device), prevs.to(device)
        optimizer.zero_grad()

        hidden = model(prevs)[1] if read_prev else None
        outputs, _ = model(inputs, hidden)
        targets = targets.view(-1)

        loss = loss_func(outputs, targets)
        y = np.vstack([outputs.argmax(1).cpu().numpy(), targets.cpu().numpy()])  # 2, seq_len * batch_size
        y = y[:, np.where(y[1, :] != pad_idx)[0]]  # 2, filtered_non_padding
        acc = accuracy_score(y[0], y[1])

        loss.backward()
        optimizer.step()

        total_count += len(targets)  # seq_len * batch_size
        epoch_loss += loss.item() * len(targets)
        non_pad_count += len(y[1])  # filtered_non_padding
        epoch_acc += acc * len(y[1])

    return epoch_loss / total_count, epoch_acc / non_pad_count


def train_model(model, filename, train_loader, val_loader, pad_idx, optim, lr=1e-3, epochs=10, read_prev=False):
    model.to(device)
    optimizer = optim(model.parameters(), lr=lr)
    min_val_loss = float('inf')
    mistakes = None

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1: 02}')

        train_loss, train_acc = train_1_epoch(model, train_loader, loss_fn, optimizer, pad_idx, read_prev=read_prev)
        print(f'\tTrain Loss: {train_loss: .3f} | Train Acc: {train_acc * 100: .2f}%')

        val_loss, val_acc, wrong_preds = evaluate(model, val_loader, loss_fn, pad_idx, read_prev=read_prev)
        print(f'\t Val. Loss: {val_loss: .3f} |  Val. Acc: {val_acc * 100: .2f}%')

        if val_loss < min_val_loss:
            mistakes = wrong_preds
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'{filename}')
            print(f'\tModel parameters saved to {filename}')

    return mistakes


def show_mistakes(mistakes, vocab, top=35):
    df = pd.DataFrame([k for k, _ in mistakes.most_common(top)], columns=['prediction', 'ground truth'])
    return df.applymap(lambda i: vocab.itos[i])
