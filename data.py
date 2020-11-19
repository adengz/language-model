from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
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
