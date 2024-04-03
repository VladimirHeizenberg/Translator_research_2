import masks

import settings

import torch

from work import positional_encoding


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.transformer = torch.nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.output_layer = torch.nn.Linear(emb_size, tgt_vocab_size)

        self.src_tok_emb = positional_encoding.TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = positional_encoding.TokenEmbedding(tgt_vocab_size, emb_size)

        self.positional_encoding = positional_encoding.PositionalEncoding(
            emb_size, dropout=dropout
        )

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ):

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        src_mask = torch.zeros(
            (src.shape[0], src.shape[0]), device=settings.DEVICE
        ).type(torch.bool)

        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )

        return self.output_layer(outs)

    def encode(self, src: torch.Tensor):
        src_mask = torch.zeros(
            (src.shape[0], src.shape[0]), device=settings.DEVICE
        ).type(torch.bool)
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor):
        tgt_mask = (
            masks.generate_square_subsequent_mask(tgt.size(0)).type(torch.bool)
        ).to(settings.DEVICE)
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )
