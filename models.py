from collections import namedtuple
from typing import Optional, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F
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
            Output with loss, filtered predictions and ground_truth
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


class NegativeSamplingModel(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, dropout: float = 0.2, pad_idx: int = Vocabulary.pad_idx):
        """
        Model trained with negative sampling.

        Args:
            vocab_size: Vocabulary size.
            embedding_dim: Word embedding dimension.
            dropout: Dropout applied on embedding. Default: 0.2
            pad_idx: Index of padding token in vocabulary.
                Default: Vocabulary.pad_idx
        """
        super(NegativeSamplingModel, self).__init__()
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.out_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx

        initrange = 0.5 / embedding_dim
        self.in_embedding.weight.data.uniform_(-initrange, initrange)
        self.in_embedding.weight.data[pad_idx].zero_()
        self.out_embedding.weight.data.uniform_(-initrange, initrange)
        self.out_embedding.weight.data[pad_idx].zero_()

    def forward(self, input_encoded: torch.LongTensor, target_encoded: torch.LongTensor,
                input_lengths: torch.LongTensor, neg_samples: torch.LongTensor, **kwargs) \
            -> Output:
        """

        Args:
            input_encoded: batch_size, pad_len
            target_encoded: batch_size, pad_len
            input_lengths: batch_size
            neg_samples: batch_size, pad_len, neg_count
            kwargs: Unused kwargs from DataLoader.

        Returns:
            Output with loss, filtered predictions and ground_truth
        """
        input_embedded = self.embed_dropout(self.in_embedding(input_encoded))  # batch_size, pad_len, embedding_dim
        hidden = _feed_rnn(self.rnn, input_embedded, input_lengths)[0]  # batch_size, pad_len, embedding_dim
        hidden = hidden.view(-1, self.embedding_dim)  # batch_size * pad_len, embedding_dim

        target_encoded = target_encoded.view(-1)  # batch_size * pad_len
        pos_embedded = self.embed_dropout(self.out_embedding(target_encoded))  # batch_size * pad_len, embedding_dim
        neg_count = neg_samples.shape[-1]
        neg_embedded = self.embed_dropout(self.out_embedding(neg_samples)).view(-1, neg_count, self.embedding_dim)
        # batch_size * pad_len, neg_count, embedding_dim

        mask = (input_encoded != self.pad_idx).view(-1)
        loss = self.loss_fn(hidden, pos_embedded, neg_embedded, mask)
        vocab_embedded = self.embed_dropout(self.out_embedding.weight.data)  # vocab_size, embedding_dim
        vocab_embedded, _ = torch.broadcast_tensors(vocab_embedded[None, :, :], torch.Tensor(hidden.shape[0], 1, 1))
        # batch_size * pad_len, vocab_size, embedding_dim
        score = self.get_score(vocab_embedded, hidden)  # batch_size * pad_len, vocab_size
        predictions = score.argmax(dim=-1).masked_select(mask)
        ground_truth = target_encoded.masked_select(mask)

        return Output(loss, predictions, ground_truth)

    @staticmethod
    def get_score(embedded: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Calculates score.

        Args:
            embedded: batch_size * pad_len, sample_count, embedding_dim
            hidden: batch_size * pad_len, embedding_dim

        Returns:
            batch_size * pad_len, sample_count
                (squeezed if sample_count=1)
        """
        return torch.bmm(embedded, hidden[:, :, None]).squeeze()

    def loss_fn(self, hidden: torch.Tensor, pos_embedded: torch.Tensor, neg_embedded: torch.Tensor,
                mask: torch.BoolTensor) -> torch.Tensor:
        """
        Negative sampling loss function.

        Args:
            hidden: batch_size * pad_len, embedding_dim
            pos_embedded: batch_size * pad_len, embedding_dim
            neg_embedded: batch_size * pad_len, neg_count, embedding_dim
            mask: batch_size * pad_len

        Returns:
            0-dim loss
        """
        pos_embedded = pos_embedded[:, None, :]  # batch_size * pad_len, 1, embedding_dim

        pos_score = self.get_score(pos_embedded, hidden)  # batch_size * pad_len
        pos_contrib = -F.logsigmoid(pos_score)
        neg_score = self.get_score(neg_embedded, hidden)  # batch_size * pad_len, neg_count
        neg_contrib = -torch.log(1 - torch.sigmoid(neg_score)).mean(dim=-1)  # batch_size * pad_len

        return torch.masked_select(pos_contrib + neg_contrib, mask).mean()
