import torch
from torchtext.utils import get_tokenizer

torch.manual_seed(0)
DEVICE = torch.device("gpu")

source_language = "de"
target_language = "en"


"""Токены"""
UNK_IDX = 0  # Токен для обозначения неизвестного слова.
PAD_IDX = 1  # Токен для паддинга.
SOS_IDX = 2  # Токен начала последовательности, start of sequnce.
EOS_IDX = 3  # Токен окончания последовательности, end of sequnce.
special_symbols = ["<unk>", "<pad>", "<sos>", "<eos>"]


"""Трансформер"""

token_transform = {}
vocab_transform = {}
token_transform[source_language] = get_tokenizer(
    "spacy",
    language="de_core_news_sm"
)
token_transform[target_language] = get_tokenizer(
    "spacy",
    language="en_core_web_sm"
)

src_vocab_size = len(vocab_transform[source_language])
tgt_vocab_size = len(vocab_transform[target_language])
emb_size = 512
nhead = 16
dim_feedforward = 512
batch_size = 128
num_encoder_layers = 3
num_decoder_layers = 3

"""Обучение"""
epochs = 50
