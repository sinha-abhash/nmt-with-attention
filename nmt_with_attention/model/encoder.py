import tensorflow as tf

from nmt_with_attention.utils import ShapeChecker


class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor: tf.keras.layers.TextVectorization, units: int):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units

        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.units, mask_zero=True)

        # The RNN layer processes those vectors sequentially.
        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(
                units, return_sequences=True, recurrent_initializer='glorot_uniform'
            )
        )

    def call(self, x):
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch s')

        x = self.embedding(x)
        shape_checker(x, 'batch s units')

        x = self.rnn(x)
        shape_checker(x, 'batch s units')

        return x

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]

        context = self.text_processor(texts).to_tensor()
        context = self(context)
        return context
