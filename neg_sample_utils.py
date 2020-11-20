import torch
import torch.nn.functional as F
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
    neg_contrib = -torch.log(1 - F.sigmoid(neg_score)).mean()

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
def evaluate(model, dataloader, vocab_tensor):
    model.eval()

    total_loss = total_acc = sample_count = 0
    for inputs, targets, _, neg_samples in dataloader:
        inputs, targets, neg_samples = inputs.to(device), targets.to(device), neg_samples.to(device)

        neg_embed, hidden, pos_embed = model(neg_samples, inputs, targets)
        loss = loss_fn(neg_embed, hidden, pos_embed)
        vocab_embed = model(vocab_tensor)[0]
        preds = get_predictions(vocab_embed, hidden).cpu()
        acc = accuracy_score(preds, targets.view(-1).cpu())

        count = preds.shape[0]
        sample_count += count
        total_loss += loss.item() * count
        total_acc += acc * count

    model.train()
    return total_loss / sample_count, total_acc / sample_count


@timeit
def train_1_epoch(model, dataloader, optimizer, vocab_tensor):
    model.train()

    total_loss = total_acc = sample_count = 0
    for inputs, targets, _, neg_samples in dataloader:
        inputs, targets, neg_samples = inputs.to(device), targets.to(device), neg_samples.to(device)
        optimizer.zero_grad()

        neg_embed, hidden, pos_embed = model(neg_samples, inputs, targets)
        loss = loss_fn(neg_embed, hidden, pos_embed)
        vocab_embed = model(vocab_tensor)[0]
        preds = get_predictions(vocab_embed, hidden).cpu()
        acc = accuracy_score(preds, targets.view(-1).cpu())

        loss.backward()
        optimizer.step()

        count = preds.shape[0]
        sample_count += count
        total_loss += loss.item() * count
        total_acc += acc * count

    return total_loss / sample_count, total_acc / sample_count


def train_model(model, filename, train_loader, val_loader, vocab_size, optim, lr=1e-3, epochs=20):
    model.to(device)
    optimizer = optim(model.parameters(), lr=lr)
    vocab_tensor = torch.arange(vocab_size).to(device)
    min_val_loss = float('inf')

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1: 02}')

        train_time, train_metrics = train_1_epoch(model, train_loader, optimizer, vocab_tensor)
        train_mins, train_secs = train_time
        print(f'\tTrain time: {train_mins}m {train_secs}s')
        train_loss, train_acc = train_metrics
        print(f'\tTrain Loss: {train_loss: .3f} | Train Acc: {train_acc * 100: .2f}%')

        val_time, val_metrics = evaluate(model, val_loader, vocab_tensor)
        val_mins, val_secs = val_time
        print(f'\t Val. time: {val_mins}m {val_secs}s')
        val_loss, val_acc = val_metrics
        print(f'\t Val. Loss: {val_loss: .3f} |  Val. Acc: {val_acc * 100: .2f}%')

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'{filename}')
            print(f'\tModel parameters saved to {filename}')
