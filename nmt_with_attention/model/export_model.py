import tensorflow as tf

from nmt_with_attention.model import Translator


class Export(tf.Module):
    def __init__(self, model: Translator):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, inputs):
        return self.model.translate(inputs)
