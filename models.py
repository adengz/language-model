from collections import namedtuple
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import Vocabulary


Output = namedtuple('Output', ['loss', 'predictions', 'ground_truth'])


def _feed_rnn(rnn_layer: nn.Module, input_tensor: torch.Tensor, lengths: torch.LongTensor,
              hidden: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = None) \
        -> Tuple[torch.Tensor, Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """

    Args:
        rnn_layer: RNN layer.
        input_tensor: batch_size, pad_len
        lengths: batch_size

    Returns:
        RNN output, final hidden state
    """
    packed_input = pack_padded_sequence(input_tensor, lengths, batch_first=True, enforce_sorted=False)
    packed_output, hidden = rnn_layer(packed_input, hidden)
    output_tensor = pad_packed_sequence(packed_output, batch_first=True)[0]
    return output_tensor, hidden


class FullVocabularyModel(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout: float = 0.2,
                 read_context: bool = False, pad_idx: int = Vocabulary.pad_idx):
        """
        Model trained with full vocabulary classification.

        Args:
            vocab_size: Vocabulary size.
            embedding_dim: Word embedding dimension.
            hidden_size: RNN dimension.
            dropout: Dropout applied on embedding. Default: 0.2
            read_context: Whether use context to initiate RNN hidden
                state. Default: False
            pad_idx: Index of padding token in vocabulary.
                Default: Vocabulary.pad_idx
        """
        super(FullVocabularyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embedding_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

        self.vocab_size = vocab_size
        self.read_context = read_context
        self.pad_idx = pad_idx

        initrange = 0.5 / embedding_dim
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.weight.data[pad_idx].zero_()

    def forward(self, input_encoded: torch.LongTensor, target_encoded: torch.LongTensor,
                input_lengths: torch.LongTensor, context_encoded: torch.LongTensor, context_lengths: torch.LongTensor,
                **kwargs) -> Output:
        """

        Args:
            input_encoded: batch_size, pad_len
            target_encoded: batch_size, pad_len
            input_lengths: batch_size
            context_encoded: batch_size, ctx_pad_len
            context_lengths: batch_size
            kwargs: Unused kwargs from DataLoader.

        Returns:
            (seq_len * batch_size, vocab_size), hidden states
        """
        hidden = None
        if self.read_context:
            ctx_embedded = self.embed_dropout(self.embedding(context_encoded))
            hidden = _feed_rnn(self.rnn, ctx_embedded, context_lengths)[1]

        input_embedded = self.embed_dropout(self.embedding(input_encoded))  # batch_size, pad_len, embedding_dim
        rnn_output = _feed_rnn(self.rnn, input_embedded, input_lengths, hidden=hidden)[0]
        # batch_size, pad_len, hidden_size
        fc_output = self.linear(rnn_output)  # batch_size, pad_len, vocab_size

        loss = self.loss_fn(fc_output.view(-1, self.vocab_size), target_encoded.view(-1))
        mask = input_encoded != self.pad_idx
        ground_truth = target_encoded.masked_select(mask)
        predictions = fc_output.argmax(dim=-1).masked_select(mask)
        return Output(loss, predictions, ground_truth)


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
