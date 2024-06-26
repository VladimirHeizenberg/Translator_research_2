import numpy as np

import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.callbacks import Model2onnx, WarmupCosineDecay

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tokenizers import CustomTokenizer

from mltu.tensorflow.transformer.utils import MaskedAccuracy, MaskedLoss
from mltu.tensorflow.transformer.callbacks import EncDecSplitCallback

from model import Transformer
from configs import ModelConfigs

configs = ModelConfigs()

en_training_data_path = "Datasets/en-ru/opus.en-ru-train.en"
en_validation_data_path = "Datasets/en-ru/opus.en-ru-dev.en"
ru_training_data_path = "Datasets/en-ru/opus.en-ru-train.ru"
ru_validation_data_path = "Datasets/en-ru/opus.en-ru-dev.ru"

def read_files(path):
    with open(path, "r", encoding="utf-8") as f:
        en_train_dataset = f.read().split("\n")[:-1]
    return en_train_dataset

en_training_data = read_files(en_training_data_path)
en_validation_data = read_files(en_validation_data_path)
ru_training_data = read_files(ru_training_data_path)
ru_validation_data = read_files(ru_validation_data_path)

max_lenght = 500
train_dataset = [[ru_sentence, en_sentence] for ru_sentence, en_sentence in zip(ru_training_data, en_training_data) if len(ru_sentence) <= max_lenght and len(en_sentence) <= max_lenght]
val_dataset = [[ru_sentence, en_sentence] for ru_sentence, en_sentence in zip(ru_validation_data, en_validation_data) if len(ru_sentence) <= max_lenght and len(en_sentence) <= max_lenght]
ru_training_data, en_training_data = zip(*train_dataset)
ru_validation_data, en_validation_data = zip(*val_dataset)

tokenizer = CustomTokenizer(char_level=True)
tokenizer.fit_on_texts(ru_training_data)
tokenizer.save(configs.model_path + "/tokenizer.json")

detokenizer = CustomTokenizer(char_level=True)
detokenizer.fit_on_texts(en_training_data)
detokenizer.save(configs.model_path + "/detokenizer.json")


def preprocess_inputs(data_batch, label_batch):
    encoder_input = np.zeros((len(data_batch), tokenizer.max_length)).astype(np.int64)
    decoder_input = np.zeros((len(label_batch), detokenizer.max_length)).astype(np.int64)
    decoder_output = np.zeros((len(label_batch), detokenizer.max_length)).astype(np.int64)

    data_batch_tokens = tokenizer.texts_to_sequences(data_batch)
    label_batch_tokens = detokenizer.texts_to_sequences(label_batch)

    for index, (data, label) in enumerate(zip(data_batch_tokens, label_batch_tokens)):
        encoder_input[index][:len(data)] = data
        decoder_input[index][:len(label)-1] = label[:-1] # Drop the [END] tokens
        decoder_output[index][:len(label)-1] = label[1:] # Drop the [START] tokens

    return (encoder_input, decoder_input), decoder_output

train_dataProvider = DataProvider(
    train_dataset, 
    batch_size=configs.batch_size, 
    batch_postprocessors=[preprocess_inputs],
    use_cache=True,
    )

val_dataProvider = DataProvider(
    val_dataset, 
    batch_size=configs.batch_size, 
    batch_postprocessors=[preprocess_inputs],
    use_cache=True,
    )

transformer = Transformer(
    num_layers=configs.num_layers,
    d_model=configs.d_model,
    num_heads=configs.num_heads,
    dff=configs.dff,
    input_vocab_size=len(tokenizer)+1,
    target_vocab_size=len(detokenizer)+1,
    dropout_rate=configs.dropout_rate,
    encoder_input_size=tokenizer.max_length,
    decoder_input_size=detokenizer.max_length
    )

transformer.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=configs.init_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

transformer.compile(
    loss=MaskedLoss(),
    optimizer=optimizer,
    metrics=[MaskedAccuracy()],
    run_eagerly=False
    )

warmupCosineDecay = WarmupCosineDecay(
    lr_after_warmup=configs.lr_after_warmup,
    final_lr=configs.final_lr,
    warmup_epochs=configs.warmup_epochs,
    decay_epochs=configs.decay_epochs,
    initial_lr=configs.init_lr,
    )
earlystopper = EarlyStopping(monitor="val_masked_accuracy", patience=5, verbose=1, mode="max")
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_masked_accuracy", verbose=1, save_best_only=True, mode="max", save_weights_only=False)
tb_callback = TensorBoard(f"{configs.model_path}/logs")
reduceLROnPlat = ReduceLROnPlateau(monitor="val_masked_accuracy", factor=0.9, min_delta=1e-10, patience=2, verbose=1, mode="max")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5", metadata={"tokenizer": tokenizer.dict(), "detokenizer": detokenizer.dict()}, save_on_epoch_end=False)
encDecSplitCallback = EncDecSplitCallback(configs.model_path, encoder_metadata={"tokenizer": tokenizer.dict()}, decoder_metadata={"detokenizer": detokenizer.dict()})

configs.save()

transformer.fit(
    train_dataProvider, 
    validation_data=val_dataProvider, 
    epochs=configs.train_epochs,
    callbacks=[
        earlystopper,
        warmupCosineDecay,
        checkpoint, 
        tb_callback, 
        reduceLROnPlat,
        model2onnx,
        encDecSplitCallback
        ]
    )