from typing import Tuple

import torch
from torch import nn


class FullVocabModel(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, padding_idx: int,
                 embed_dropout: float = 0.5, rnn_dropout: float = 0.5):
        """
        Model trained with full vocabulary classification.

        Args:
            vocab_size: Vocabulary size.
            embedding_dim: Word embedding dimension.
            hidden_size: RNN dimension.
            padding_idx: Index of padding token in vocabulary.
            embed_dropout: Embedding dropout. Default: 0.5
            rnn_dropout: RNN dropout. Default: 0.5
        """
        super(FullVocabModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.rnn = nn.LSTM(embedding_dim, hidden_size)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

        initrange = 0.5 / embedding_dim
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.weight.data[padding_idx].zero_()

    def forward(self, text: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        Args:
            text: seq_len, batch_size
            hidden: Hidden states generated by previous context
                Usage:
                $ context_hidden = model(prev)[1]
                $ output, _ = model(text, context_hidden)

        Returns:
            (seq_len * batch_size, vocab_size), hidden states
        """
        embedded = self.embed_dropout(self.embedding(text))  # seq_len, batch_size, embedding_dim
        if hidden is None:  # w/o prev
            rnn_output, hidden = self.rnn(embedded)
        else:  # w/ prev
            rnn_output, hidden = self.rnn(embedded, hidden)
        rnn_output = self.rnn_dropout(rnn_output)  # seq_len, batch_size, hidden_size
        fc_output = self.linear(rnn_output)  # seq_len, batch_size, vocab_size
        return fc_output.view(-1, self.vocab_size), hidden  # seq_len * batch_size, vocab_size


class NegSampleModel(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int,
                 embed_dropout: float = 0.5, rnn_dropout: float = 0.5):
        """
        Model trained with negative sampling.

        Args:
            vocab_size: Vocabulary size.
            embedding_dim: Word embedding dimension.
            padding_idx: Index of padding token in vocabulary.
            embed_dropout: Embedding dropout. Default: 0.5
            rnn_dropout: RNN dropout. Default: 0.5
        """
        super(NegSampleModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.rnn = nn.LSTM(embedding_dim, embedding_dim)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.embedding_dim = embedding_dim

        initrange = 0.5 / embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.in_embed.weight.data[padding_idx].zero_()
        self.out_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data[padding_idx].zero_()

    def forward(self, samples: torch.Tensor, text: torch.Tensor = None, targets: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            samples: seq_len, batch_size, sample_size
                or vocab_size
            text: seq_len, batch_size
                (Default to None when only embedding vocabulary)
            targets: seq_len, batch_size
                (Default to None when only embedding vocabulary)

        Returns:
            (seq_len * batch_size, sample_size, embedding_dim),
            (seq_len * batch_size, embedding_dim, 1)
                or None when only embedding vocabulary,
            (seq_len * batch_size, 1, embedding_dim)
                or None when only embedding vocabulary
        """
        sample_size = samples.shape[-1]
        samples_embedded = self.embed_dropout(self.out_embed(samples))
        # seq_len, batch_size, sample_size, embedding_dim
        samples_embedded = samples_embedded.view(-1, sample_size, self.embedding_dim)
        # seq_len * batch_size, sample_size, embedding_dim

        rnn_output = None
        if text is not None:
            text_embedded = self.embed_dropout(self.in_embed(text))  # seq_len, batch_size, embedding_dim
            rnn_output = self.rnn_dropout(self.rnn(text_embedded)[0])  # seq_len, batch_size, embedding_dim
            rnn_output = rnn_output.view(-1, self.embedding_dim).unsqueeze(2)  # seq_len * batch_size, embedding_dim, 1

        targets_embedded = None
        if targets is not None:
            targets_embedded = self.embed_dropout(self.out_embed(targets))  # seq_len, batch_size, embedding_dim
            targets_embedded = targets_embedded.view(-1, self.embedding_dim).unsqueeze(1)
            # seq_len * batch_size, 1, embedding_dim

        return samples_embedded, rnn_output, targets_embedded
