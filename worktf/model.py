import tensorflow as tf

from mltu.tensorflow.transformer.layers import Encoder, Decoder

def Transformer(
    input_vocab_size: int, 
    target_vocab_size: int, 
    encoder_input_size: int = None,
    decoder_input_size: int = None,
    num_layers: int=6, 
    d_model: int=512, 
    num_heads: int=8,
    dff: int=2048,
    dropout_rate: float=0.1,
    ):
    inputs = [
        tf.keras.layers.Input(shape=(encoder_input_size,), dtype=tf.int64), 
        tf.keras.layers.Input(shape=(decoder_input_size,), dtype=tf.int64)
        ]
    
    encoder_input, decoder_input = inputs

    encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=input_vocab_size, dropout_rate=dropout_rate)(encoder_input)
    decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=target_vocab_size, dropout_rate=dropout_rate)(decoder_input, encoder)

    output = tf.keras.layers.Dense(target_vocab_size)(decoder)

    return tf.keras.Model(inputs=inputs, outputs=output)