from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import einops

from nmt_with_attention.model import Encoder, Decoder
from nmt_with_attention.utils import tf_lower_and_split_punct, ShapeChecker


class Translator(tf.keras.Model):
    def __init__(self, units, context_text_processor, target_text_processor):
        super().__init__()
        self.encoder = Encoder(context_text_processor, units)
        self.decoder = Decoder(target_text_processor, units)
        self.last_attention_weights = None

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits

    def translate(self, texts, *, max_length=50, temperature=0.0):
        shape_checker = ShapeChecker()
        context = self.encoder.convert_input(texts)
        shape_checker(context, "batch s units")

        tokens = []
        attention_weights = []
        next_token, done, state = self.decoder.get_initial_state(context)

        tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)

        for t in tf.range(max_length):
            next_token, done, state = self.decoder.get_next_token(
                context, next_token, done, state, temperature
            )

            tokens = tokens.write(t, next_token)
            attention_weights.append(self.decoder.last_attention_weights)

            if tf.reduce_all(done):
                break

        tokens = tokens.stack()
        shape_checker(tokens, "t batch t1")
        tokens = einops.rearrange(tokens, "t batch 1 -> batch t")
        shape_checker(tokens, "batch t")

        result = self.decoder.tokens_to_text(tokens)
        shape_checker(result, "batch")
        return result

    def plot_attention(self, text: str, **kwargs):
        output = self.translate([text], **kwargs)
        output = output[0].numpy().decode()

        attention = self.last_attention_weights[0]

        context: tf.Tensor = tf_lower_and_split_punct(text)
        context = context.numpy().decode().split()

        output: tf.Tensor = tf_lower_and_split_punct(output)
        output = output.numpy().decode().split()[1:]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap="viridis", vmin=0.0)
        fontdict = {"fontsize": 14}
        ax.set_xticklabels([""] + context, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([""] + output, fontdict=fontdict)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        ax.set_xlabel("Input text")
        ax.set_ylabel("Output text")
