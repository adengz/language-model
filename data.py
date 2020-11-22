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
            vocab_fname: Vocabulary filename in DATA_ROOT.
            train_fname: Training data filename in DATA_ROOT.
        """
        self.itos = [self.pad_token]
        with open(DATA_ROOT / vocab_fname) as f:
            self.itos += [line.strip() for line in f.readlines()]
        self.stoi = {word: i for i, word in enumerate(self.itos)}

        with open(DATA_ROOT / train_fname) as f:
            counter = Counter(f.read().replace('\n', ' ').split(' '))
        self.counts = [0] + [counter[w] for w in self.itos[1:]]

    def __len__(self) -> int:
        return len(self.itos)


class BobSueDataset(Dataset):

    def __init__(self, filename: str, vocab: Vocabulary, neg_count: int = 0, sample_pow: float = 0.):
        """

        Args:
            filename: Dataset filename in DATA_ROOT.
            vocab: Vocabulary.
            neg_count: Number of negative samples for each word.
                Default: 0
            sample_pow: Power applied to occurrence. Default: 0
        """
        super(BobSueDataset, self).__init__()
        self.df = pd.read_csv(DATA_ROOT / filename, sep='\t', header=None)
        self.vocab = vocab
        self.neg_count = neg_count
        self.freqs = torch.Tensor(vocab.counts) ** sample_pow

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """

        Args:
            idx: Index.

        Returns:
            input_encoded: seq_len - 1
            target_encoded: seq_len - 1
            prev_encoded: prev_len
            neg_samples: seq_len - 1, neg_count
        """
        prev, text = self.df.iloc[idx]
        prev_encoded = torch.LongTensor([self.vocab.stoi[w] for w in prev.split()])
        text_encoded = torch.LongTensor([self.vocab.stoi[w] for w in text.split()])
        input_encoded = text_encoded[:-1]
        target_encoded = text_encoded[1:]

        neg_samples = []
        if self.neg_count:
            for te in target_encoded:
                freqs = torch.clone(self.freqs)
                freqs[te] = 0
                neg_samples.append(torch.multinomial(freqs, self.neg_count, True).tolist())

        return input_encoded, target_encoded, prev_encoded, torch.LongTensor(neg_samples)


class PadSeqCollate:

    def __init__(self, pad_idx: int):
        """

        Args:
            pad_idx: Padding index in vocabulary.
        """
        self.pad_idx = pad_idx

    def __call__(self, batch: Sequence[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]]) \
            -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        inputs, targets, prevs, negs = zip(*batch)
        return self.pad_sequence(inputs), self.pad_sequence(targets), \
               self.pad_sequence(prevs, pad_left=True), self.pad_sequence(negs)

    def pad_sequence(self, sequences: Sequence[torch.LongTensor], pad_left: bool = False) -> torch.LongTensor:
        """
        Pads sequences to the same length with pad_idx.

        Args:
            sequences: torch.LongTensor with various first dimension.
            pad_left: Whether use pre-sequence (left) padding or
                post-sequence padding (right). Default: False

        Returns:
            len(sequences), max_seq_len
        """
        final_len = max([s.shape[0] for s in sequences])
        trailing_dims = sequences[0].shape[1:]
        out_dims = (final_len, len(sequences)) + trailing_dims
        out_tensor = sequences[0].new_full(out_dims, self.pad_idx)

        for i, tensor in enumerate(sequences):
            length = tensor.shape[0]
            if pad_left:
                out_tensor[-length:, i, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor

        return out_tensor


def get_dataloader(filename: str, vocab: Vocabulary, batch_size: int, neg_count: int = 0, sample_pow: float = 0.,
                   shuffle: bool = True, pin_memory: bool = True) -> DataLoader:
    """
    Wrapper function for creating a DataLoader loading a BobSueDataset.

    Args:
        filename: Dataset filename in DATA_ROOT.
        vocab: Vocabulary.
        batch_size: Batch size.
        neg_count: See BobSueDataset docs.
        sample_pow: See BobSueDataset docs.
        shuffle: Whether reshuffle data at each epoch, see DataLoader
            docs. Default: True
        pin_memory: Whether use pinned memory, see DataLoader docs.
            Default: True

    Returns:
        DataLoader.
    """
    dataset = BobSueDataset(filename, vocab, neg_count, sample_pow)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=PadSeqCollate(vocab.pad_idx),
                        shuffle=shuffle, pin_memory=pin_memory)
    return loader
