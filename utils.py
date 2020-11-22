from typing import Callable, Any, Tuple
import time
from collections import Counter

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from sklearn.metrics import accuracy_score
import pandas as pd

from models import FullVocabModel, NegSampleModel
from data import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def timeit(f: Callable) -> Any:
    """
    Timer decorator for profiling.

    Args:
        f: Decorated function.

    Returns:
        Whatever returned by invoking f.
    """
    def timed(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        print(f'\t Wall Time: {elapsed:.3f} s')

        return ret
    return timed


def neg_sample_loss(neg_embed: torch.Tensor, hidden: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
    """
    Loss function for negative sampling model.

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


def get_loss_function(model: nn.Module) -> Callable[..., torch.Tensor]:
    """
    Retrieves loss function for specific model.

    Args:
        model: Model.

    Returns:
        Loss function.
    """
    if isinstance(model, FullVocabModel):
        return nn.CrossEntropyLoss()
    elif isinstance(model, NegSampleModel):
        return neg_sample_loss
    else:
        raise NotImplementedError


def get_metrics(model: nn.Module,
                batch: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor],
                loss_fn: Callable[..., torch.Tensor], pad_idx: int, return_mismatch: bool = False,
                read_prev: bool = False, vocab_tensor: torch.LongTensor = None) \
        -> Tuple[torch.Tensor, int, float, int, Counter]:
    """
    Passes one batch of data to model and returns a bunch of metrics.

    Args:
        model: Model.
        batch: One batch data from iterating DataLoader.
        loss_fn: Loss function.
        pad_idx: Padding index, for artificial data removal.
        return_mismatch: Whether return a Counter of mismatched
            prediction and ground truth pair. Default: False
        read_prev: Whether read previous sentence. Used by
            FullVocabModel only. Default: False
        vocab_tensor: Vocabulary tensor. Used by NegSampleModel only.
            Default: None

    Returns:
        loss: 0-dim
        count: Total output count, seq_len * batch_size
        acc: Effective accuracy on non padding tokens
        non_pads: Total non padding tokens count
        counter: Counter of mismatches between prediction and ground
            truth
    """
    inputs, targets, prevs, negs = batch
    inputs, targets, prevs, negs = inputs.to(device), targets.to(device), prevs.to(device), negs.to(device)

    if isinstance(model, FullVocabModel):
        hidden = model(prevs)[1] if read_prev else None
        outputs = model(inputs, hidden)[0]  # seq_len * batch_size, vocab_size
        loss = loss_fn(outputs, targets.view(-1))
        prediction = outputs.argmax(1)  # seq_len * batch_size
    elif isinstance(model, NegSampleModel):
        neg_embed, hidden, pos_embed = model(negs, inputs, targets)  # seq_len * batch_size, embedding_dim, 1
        loss = loss_fn(neg_embed, hidden, pos_embed)
        vocab_embed = model(vocab_tensor)[0]  # 1, vocab_size, embedding_dim
        vocab_embed, _ = torch.broadcast_tensors(vocab_embed, torch.Tensor(hidden.shape[0], 1, 1))
        # seq_len * batch_size, vocab_size, embedding_dim
        scores = torch.bmm(vocab_embed, hidden).squeeze()  # seq_len * batch_size, vocab_size
        prediction = scores.argmax(1)  # seq_len * batch_size
    else:
        raise NotImplementedError

    y = torch.vstack([prediction, targets.view(-1)]).cpu()  # 2, seq_len * batch_size
    y = y[:, torch.where(y[1, :] != pad_idx)[0]]  # 2, filtered_non_pads
    acc = accuracy_score(y[0], y[1])
    counter = None
    if return_mismatch:
        mismatches = y[:, torch.where(y[0, :] != y[1, :])[0]]  # 2, filtered_wrong_preds
        counter = Counter(map(tuple, mismatches.t().tolist()))
    return loss, len(prediction), acc, len(y[0]), counter


@timeit
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, loss_fn: Callable[..., torch.Tensor], **kwargs: Any) \
        -> Tuple[float, float, Counter]:
    """
    Evaluates model with a given dataset.

    Args:
        model: Model.
        dataloader: DataLoader.
        loss_fn: Loss function
        **kwargs: kwargs supported by get_metrics.

    Returns:
        Loss, accuracy, mismatch counter
    """
    model.eval()

    epoch_loss = epoch_acc = total_count = non_pad_count = 0
    epoch_mismatches = Counter()
    for batch in dataloader:
        loss, count, acc, non_pads, mismatches = get_metrics(model, batch, loss_fn, return_mismatch=True, **kwargs)

        total_count += count
        epoch_loss += loss.item() * count
        non_pad_count += non_pads
        epoch_acc += acc * non_pads
        epoch_mismatches += mismatches

    model.train()
    return epoch_loss / total_count, epoch_acc / non_pad_count, epoch_mismatches


@timeit
def train_1_epoch(model: nn.Module, dataloader: DataLoader, loss_fn: Callable[..., torch.Tensor], optimizer: Optimizer,
                  **kwargs: Any) -> Tuple[float, float]:
    """
    Trains model by 1 epoch.

    Args:
        model: Model.
        dataloader: DataLoader of training dataset.
        loss_fn: Loss function.
        optimizer: Optimizer.
        **kwargs: kwargs supported by get_metrics.

    Returns:
        Loss, accuracy
    """
    model.train()

    epoch_loss = epoch_acc = total_count = non_pad_count = 0
    for batch in dataloader:
        optimizer.zero_grad()
        loss, count, acc, non_pads, _ = get_metrics(model, batch, loss_fn, **kwargs)

        loss.backward()
        optimizer.step()

        total_count += count
        epoch_loss += loss.item() * count
        non_pad_count += non_pads
        epoch_acc += acc * non_pads

    return epoch_loss / total_count, epoch_acc / non_pad_count


def train_model(model: nn.Module, filename: str, train_loader: DataLoader, valid_loader: DataLoader,
                optim: Callable[..., Optimizer], lr: float = 1e-3, epochs: int = 1, vocab_size: int = None,
                **kwargs: Any) -> Counter:
    """
    Train a model, while saving the parameters of model with lowest
    validation loss.

    Args:
        model: Model.
        filename: Filename to save model parameters.
        train_loader: DataLoader for training dataset.
        valid_loader: DataLoader for validation dataset.
        optim: Optimizer class.
        lr: Learning rate. Default: 1e-3
        epochs: Number of epochs. Default: 1
        vocab_size: Vocabulary size, used by NegSampleModel only.
            Default: None
        **kwargs: kwargs supported by get_metrics.

    Returns:
        Mismatch counter of the best model.
    """
    if isinstance(model, NegSampleModel):
        assert vocab_size is not None, 'Vocabulary size is a non-optional arg when training negative sampling model'
        kwargs.update({'vocab_tensor': torch.arange(vocab_size).to(device)})
    model.to(device)
    loss_fn = get_loss_function(model)
    optimizer = optim(model.parameters(), lr=lr)
    min_valid_loss = float('inf')
    mismatches = None

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1:02}')

        train_loss, train_acc = train_1_epoch(model, train_loader, loss_fn, optimizer, **kwargs)
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

        valid_loss, valid_acc, epoch_mismatches = evaluate(model, valid_loader, loss_fn, **kwargs)
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}%')

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            mismatches = epoch_mismatches
            torch.save(model.state_dict(), f'{filename}')
            print(f'\tModel parameters saved to {filename}')
        else:
            print()

    return mismatches


def show_mistakes(mismatches: Counter, vocab: Vocabulary, top: int = 35) -> pd.DataFrame:
    """
    List top prediction mistakes.

    Args:
        mismatches: Mismatch counter from model evaluation.
        vocab: Vocabulary.
        top: Number of mismatches to list. Default: 35

    Returns:
        DataFrame with wrongly predicted words and corresponding
        ground truth sorted by occurrence from high to low.
    """
    df = pd.DataFrame([k for k, _ in mismatches.most_common(top)], columns=['prediction', 'ground truth'])
    return df.applymap(lambda i: vocab.itos[i])
