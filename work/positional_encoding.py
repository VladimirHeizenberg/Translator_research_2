import math

import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        den = torch.pow(10000.0, torch.arange(0, emb_size, 2) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)

        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos / den)
        pos_embedding[:, 1::2] = torch.cos(pos / den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
