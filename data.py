from pathlib import Path
from collections import Counter
from typing import Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

DATA_ROOT = Path('data')


class Vocabulary:

    pad_token = '<pad>'
    pad_idx = 0

    def __init__(self, vocab_fname: str = 'bobsue.voc.txt', train_fname: str = 'bobsue.lm.train.txt'):
        """

        Args:
            vocab_fname: Vocabulary filename in DATA_ROOT
            train_fname: Training data filename in DATA_ROOT
        """
        self.itos = [self.pad_token]
        with open(DATA_ROOT / vocab_fname) as f:
            self.itos += [line.strip() for line in f.readlines()]
        self.stoi = {word: i for i, word in enumerate(self.itos)}

        with open(DATA_ROOT / train_fname) as f:
            counter = Counter(f.read().replace('\n', ' ').split(' '))
        self.counts = [0] + [counter[w] for w in self.itos]
    
    def __len__(self):
        return len(self.itos)


class BobSueDataset(Dataset):

    def __init__(self, filename: str, vocab: Vocabulary, neg_count: int = 0, sample_pow: float = 1.):
        """

        Args:
            filename: Dataset filename in DATA_ROOT
            vocab: Vocabulary
            neg_count: Number of negative samples for each word
            sample_pow: Power applied to sample frequency
        """
        super(BobSueDataset, self).__init__()
        self.df = pd.read_csv(DATA_ROOT / filename, sep='\t', header=None)
        self.vocab = vocab
        self.neg_count = neg_count
        self.freqs = torch.Tensor(vocab.counts) ** sample_pow

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        prev, text = self.df.iloc[idx]
        prev_encoded = torch.LongTensor([self.vocab.stoi[w] for w in prev.split()])
        text_encoded = torch.LongTensor([self.vocab.stoi[w] for w in text.split()])

        neg_samples = []
        if self.neg_count:
            for te in text_encoded:
                freqs = torch.clone(self.freqs)
                freqs[te] = 0
                neg_samples.append(torch.multinomial(freqs, self.neg_count, True))

        return text_encoded, prev_encoded, torch.stack(neg_samples) if neg_samples else None


def pad_sequence(sequences: Sequence[torch.Tensor], pad_value: int, pad_left: bool = False):
    """
    An alternative implementation of torch.nn.utils.rnn.pad_sequence
    with pre-sequence (left) padding feature.

    Args:
        sequences: torch.Tensor with various first dimension
        pad_value: Padding value
        pad_left: Pre-sequence (left) padding or post-sequence (right)
            padding. Default: False

    Returns:

    """
    final_len = max([s.shape[0] for s in sequences])
    trailing_dims = sequences[0].shape[1:]
    out_dims = (final_len, len(sequences)) + trailing_dims
    out_tensor = sequences[0].new_full(out_dims, pad_value)

    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        if pad_left:
            out_tensor[-length:, i, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


class PadSeqCollate:

    def __init__(self, pad_idx: int):
        """

        Args:
            pad_idx: Padding index in vocabulary
        """
        self.pad_idx = pad_idx

    def __call__(self, batch: Sequence[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]]):
        texts, prevs, negs = zip(*batch)
        batch_texts = pad_sequence(texts, self.pad_idx)
        batch_prevs = pad_sequence(prevs, self.pad_idx, pad_left=True)
        batch_negs = pad_sequence(negs, self.pad_idx) if negs[0] is not None else None
        return batch_texts, batch_prevs, batch_negs


def get_dataloader(filename: str, vocab: Vocabulary, batch_size: int, neg_count: int = 0, sample_pow: float = 1.,
                   shuffle: bool = True, pin_memory: bool = True):
    """
    Wrapper function for creating a DataLoader loading a BobSueDataset.

    Args:
        filename: Dataset filename
        vocab: Vocabulary
        batch_size: Batch size
        neg_count: See BobSueDataset
        sample_pow: See BobSueDataset
        shuffle: Whether reshuffle data at each epoch, see DataLoader
            docs. Default: True
        pin_memory: Whether use pinned memory, see DataLoader docs.
            Default: True

    Returns:
        DataLoader
    """
    dataset = BobSueDataset(filename, vocab, neg_count, sample_pow)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=PadSeqCollate(vocab.pad_idx),
                        shuffle=shuffle, pin_memory=pin_memory)
    return loader
