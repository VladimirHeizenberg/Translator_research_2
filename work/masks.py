import settings

import torch


# Матрица
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=settings.DEVICE)) == 1).transpose(
        0, 1
    )
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    # src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    # Создаем маски для входной и выходной последовательностей.
    # Для входной последовательности она не будет ничего делать.
    # То есть будет заполнена 0.
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)

    # Создаем маски для игнонироваания [PAD].
    src_padding_mask = (src == settings.PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == settings.PAD_IDX).transpose(0, 1)
    return tgt_mask, src_padding_mask, tgt_padding_mask
