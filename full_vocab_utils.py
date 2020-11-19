import time

import torch
from torch import nn
from sklearn.metrics import accuracy_score

from utils import calculate_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, dataloader, loss_func, read_prev=False):
    model.eval()

    total_loss = total_acc = sample_count = 0
    for inputs, targets, prevs, _ in dataloader:
        inputs, targets, prevs = inputs.to(device), targets.to(device), prevs.to(device)

        hidden = model(prevs)[1] if read_prev else None
        outputs, _ = model(inputs, hidden)
        targets = targets.view(-1)

        loss = loss_func(outputs, targets)
        preds = outputs.argmax(1)
        acc = accuracy_score(preds.cpu(), targets.cpu())

        count = len(targets)
        sample_count += count
        total_loss += loss.item() * count
        total_acc += acc * count

    model.train()
    return total_loss / sample_count, total_acc / sample_count


def train_1_epoch(model, dataloader, loss_func, optimizer, read_prev=False):
    model.train()

    total_loss = total_acc = sample_count = 0
    for inputs, targets, prevs, _ in dataloader:
        inputs, targets, prevs = inputs.to(device), targets.to(device), prevs.to(device)
        optimizer.zero_grad()

        hidden = model(prevs)[1] if read_prev else None
        outputs, _ = model(inputs, hidden)
        targets = targets.view(-1)

        loss = loss_func(outputs, targets)
        preds = outputs.argmax(1)
        acc = accuracy_score(preds.cpu(), targets.cpu())

        loss.backward()
        optimizer.step()

        count = len(targets)
        sample_count += count
        total_loss += loss.item() * count
        total_acc += acc * count

    return total_loss / sample_count, total_acc / sample_count


def train_model(model, filename, train_loader, val_loader, optim, lr=1e-3, epochs=10, read_prev=False):
    model.to(device)
    optimizer = optim(model.parameters(), lr=lr)
    min_val_loss = float('inf')

    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train_1_epoch(model, train_loader, loss_fn, optimizer, read_prev=read_prev)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, read_prev=read_prev)
        end = time.time()
        mins, secs = calculate_time(start, end)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'{filename}')

        print(f'Epoch: {epoch + 1: 02} | Epoch Time: {mins}m {secs}s')
        print(f'\tTrain Loss: {train_loss: .3f} | Train Acc: {train_acc * 100: .2f}%')
        print(f'\t Val. Loss: {val_loss: .3f} |  Val. Acc: {val_acc * 100: .2f}%')