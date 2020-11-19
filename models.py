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
            embed_dropout: Embedding dropout. Default: 0.5
            rnn_dropout: RNN dropout. Default: 0.5
        """
        super(FullVocabModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.rnn = nn.LSTM(embedding_dim, hidden_size)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, text: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:  # seq_len, batch_size
        embedded = self.embed_dropout(self.embedding(text))  # seq_len, batch_size, embedding_dim
        rnn_output, hidden = self.rnn(embedded)
        rnn_output = self.rnn_dropout(rnn_output)  # seq_len, batch_size, hidden_size
        fc_output = self.linear(rnn_output)  # seq_len, batch_size, vocab_size
        return fc_output, hidden