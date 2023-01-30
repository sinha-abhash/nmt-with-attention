import tensorflow as tf

from nmt_with_attention.utils import ShapeChecker
from nmt_with_attention.model import CrossAttention


class Decoder(tf.keras.layers.Layer):
    def __init__(self, text_processor: tf.keras.layers.TextVectorization, units: int):
        super(Decoder, self).__init__()
        self.last_attention_weights = None
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]'
        )
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]', invert=True
        )
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')

        self.units = units

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)

        self.rnn = tf.keras.layers.GRU(units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.attention = CrossAttention(units)

        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self, context, x, state=None, return_state=False):
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch t')
        shape_checker(context, 'batch s units')

        x = self.embedding(x)
        shape_checker(x, 'batch t units')

        x, state = self.rnn(x, initial_state=state)
        shape_checker(x, 'batch t units')

        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights
        shape_checker(x, 'batch t units')
        shape_checker(self.last_attention_weights, 'batch t s')

        logits = self.output_layer(x)
        shape_checker(logits, 'batch t target_vocab_size')

        if return_state:
            return logits, state
        else:
            return logits

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(start_tokens)
        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):
        words = self.id_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
        result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
        return result

    def get_next_token(self, context, next_token, done, state, temperature=0.0):
        logits, state = self(
            context, next_token, state=state, return_state=True
        )

        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :] / temperature
            next_token = tf.random.categorical(logits, num_samples=1)

        done = done | (next_token == self.end_token)

        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

        return next_token, done, state
