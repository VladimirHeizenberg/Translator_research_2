import settings
import torch
from torchtext.utils import get_tokenizer
from torch.utils.data import DataLoader
import transfromer
from typing import Iterable, List
import masks

transfromer_model = transfromer.Seq2SeqTransformer(
    settings.num_encoder_layers,
    settings.num_decoder_layers,
    settings.emb_size,
    settings.nhead,
    settings.src_vocab_size,
    settings.tgt_vocab_size,
    settings.dim_feedforward,
)

for p in transfromer_model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

transfromer_model = transfromer_model.to(settings.DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=settings.PAD_IDX)

optimizer = torch.optim.Adam(
    transfromer_model.parameters(),
    lr=0.0001, betas=(0.9, 0.98),
    eps=1e-9
)


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([settings.SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([settings.EOS_IDX])))


text_transform = {}
for ln in [settings.source_language, settings.target_language]:
    text_transform[ln] = sequential_transforms(
        settings.token_transform[ln],
        settings.vocab_transform[ln],
        tensor_transform
    )


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(
            text_transform[settings.source_language](src_sample.rstrip("\n"))
        )
        tgt_batch.append(
            text_transform[settings.target_language](tgt_sample.rstrip("\n"))
        )

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch,
        padding_value=settings.PAD_IDX
    )
    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch,
        padding_value=settings.PAD_IDX
    )
    return src_batch, tgt_batch


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(
        split='train',
        language_pair=(settings.source_language, settings.target_language)
    )
    train_dataloader = DataLoader(
        train_iter,
        batch_size=settings.batch_size,
        collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(settings.DEVICE)
        tgt = tgt.to(settings.DEVICE)
        tgt_input = tgt[:-1, :]
        tgt_mask, src_padding_mask, tgt_padding_mask = masks.create_mask(
            src,
            tgt_input
        )

        logits = model(
            src,
            tgt_input,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask
        )
        logits = logits.reshape(-1, logits.shape[-1])
        optimizer.zero_grad()

        tgt_out = tgt[1:, :].reshape(-1)

        loss = loss_fn(logits, tgt_out)
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(
        settings.source_language,
        settings.target_language)
        )
    val_dataloader = DataLoader(
        val_iter,
        batch_size=settings.batch_size,
        collate_fn=collate_fn
        )

    for src, tgt in val_dataloader:
        src = src.to(settings.DEVICE)
        tgt = tgt.to(settings.DEVICE)

        tgt_input = tgt[:-1, :]

        tgt_mask, src_padding_mask, tgt_padding_mask = masks.create_mask(src, tgt_input)

        logits = model(
            src,
            tgt_input,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask
        )

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(
            -1,
            logits.shape[-1]),
            tgt_out.reshape(-1)
        )
        losses += loss.item()

    return losses / len(list(val_dataloader))


