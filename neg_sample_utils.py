import torch
import torch.nn.functional as F


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
