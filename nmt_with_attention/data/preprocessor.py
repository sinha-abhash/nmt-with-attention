import logging
from typing import Tuple
import tensorflow as tf
import tensorflow_text as tf_text

from nmt_with_attention.config import MAX_VOCAB_SIZE
from nmt_with_attention.utils import tf_lower_and_split_punct


class Preprocessor:
    def __init__(self, train: tf.Tensor, val: tf.Tensor):
        self.train = train
        self.val = val
        self.logger = logging.getLogger(__name__)

        self.context_text_preprocessor = tf.keras.layers.TextVectorization(
            standardize=tf_lower_and_split_punct, max_tokens=MAX_VOCAB_SIZE, ragged=True
        )

        self.target_text_preprocessor = tf.keras.layers.TextVectorization(
            standardize=tf_lower_and_split_punct, max_tokens=MAX_VOCAB_SIZE, ragged=True
        )
        self.logger.info("Vectorizing data")
        self.vectorization()

    def vectorization(self) -> None:
        self.context_text_preprocessor.adapt(
            self.train.map(lambda context, target: context)
        )
        self.target_text_preprocessor.adapt(
            self.train.map(lambda context, target: target)
        )

    def process_text(
        self, context: tf.Tensor, target: tf.Tensor
    ) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """Convert tf.data.Datasets to zero padded tensors of token IDs.
        Also, convert context, target pair to (context, target_in), target_out pair.
        target_out is result of shifting target_in by 1 position.

        Return:
        ======
        (context, target_in), target_out
        """
        context = self.context_text_preprocessor(context).to_tensor()

        target = self.target_text_preprocessor(target)
        target_in = target[:, :-1].to_tensor()
        target_out = target[:, 1:].to_tensor()

        return (context, target_in), target_out
