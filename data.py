from pathlib import Path
from collections import Counter
from functools import partial
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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

    def tokenize(self, sentence: str) -> List[int]:
        """
        Tokenize input sentence.

        Args:
            sentence: Input sentence

        Returns:
            List of tokens representing sentence.
        """
        return [self.stoi[w] for w in sentence.split(' ')]


class BobSueDataset(Dataset):

    def __init__(self, filename: str, vocab: Vocabulary):
        """

        Args:
            filename: Dataset filename in DATA_ROOT.
            vocab: Vocabulary.
        """
        self.text_encodings, self.context_encodings = [], []
        with open(DATA_ROOT / filename) as f:
            for line in f:
                context, text = line.rstrip('\n').split('\t')
                self.text_encodings.append(vocab.tokenize(text))
                self.context_encodings.append(vocab.tokenize(context))
        self.freqs = torch.Tensor(vocab.counts)
        self._neg_count = 0
        self._sample_pow = 0.

    @property
    def neg_count(self) -> int:
        """
        Number of negative samples for each word. Default: 0
        """
        return self._neg_count

    @neg_count.setter
    def neg_count(self, neg_count: int):
        if neg_count < 0:
            raise ValueError('Expecting negative sample count greater than or equal to 0.')
        self._neg_count = neg_count

    @property
    def sample_pow(self) -> float:
        """
        Power applied to occurrence when performing sampling.
            Default: 0.
        """
        return self._sample_pow

    @sample_pow.setter
    def sample_pow(self, sample_pow: float):
        self._sample_pow = sample_pow

    def __len__(self) -> int:
        return len(self.text_encodings)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """

        Args:
            idx: Index.

        Returns:
            input_encoded: seq_len - 1
            target_encoded: seq_len - 1
            context_encoded: ctx_len
            neg_samples: seq_len - 1, neg_count
        """
        text_encoded = self.text_encodings[idx]
        input_encoded = torch.LongTensor(text_encoded[:-1])
        target_encoded = torch.LongTensor(text_encoded[1:])
        context_encoded = torch.LongTensor(self.context_encodings[idx])

        neg_samples = []
        if self.neg_count:
            for te in target_encoded:
                freqs = self.freqs ** self.sample_pow
                freqs[te] = 0
                neg_samples.append(torch.multinomial(freqs, self.neg_count, replacement=True).tolist())

        return input_encoded, target_encoded, context_encoded, torch.LongTensor(neg_samples)


pad_zeros = partial(pad_sequence, batch_first=True, padding_value=0)


def padding_collate(batch: List[Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]]) \
        -> Dict[str, torch.LongTensor]:
    """
    Collate function bridging BobSueDataset and all models.

    Args:
        batch: Batch of data from BobSueDataset.

    Returns:
        Dict with padded tensors as values.
    """
    input_encoded, target_encoded, context_encoded, neg_samples = zip(*batch)
    input_lengths = [len(ie) for ie in input_encoded]
    context_lengths = [len(ce) for ce in context_encoded]
    return {'input_encoded': pad_zeros(input_encoded), 'target_encoded': pad_zeros(target_encoded),
            'input_lengths': torch.LongTensor(input_lengths), 'context_encoded': pad_zeros(context_encoded),
            'context_lengths': torch.LongTensor(context_lengths), 'neg_samples': pad_zeros(neg_samples)}


def get_dataloader(dataset: BobSueDataset, batch_size: int, shuffle: bool = True, pin_memory: bool = True) \
        -> DataLoader:
    """
    Wrapper function for creating a DataLoader loading a BobSueDataset.

    Args:
        dataset: BobSueDataset.
        batch_size: Batch size.
        shuffle: Whether reshuffle data at each epoch, see DataLoader
            docs. Default: True
        pin_memory: Whether use pinned memory, see DataLoader docs.
            Default: True

    Returns:
        DataLoader.
    """
    return DataLoader(dataset, batch_size=batch_size, collate_fn=padding_collate,
                      shuffle=shuffle, pin_memory=pin_memory)
