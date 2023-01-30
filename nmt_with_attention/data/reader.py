from pathlib import Path
import numpy as np
import logging
from typing import Tuple

import tensorflow as tf


class DataReader:
    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Given path does not exists or not a file: {str(self.path)}"
            )

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        self.logger.info("Reading path: %s", str(self.path))
        text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines()
        pairs = [line.split("\t") for line in lines]
        context = np.array([context1 for target, context1, context2 in pairs])
        target = np.array([target for target, context1, context2 in pairs])

        return target, context

    def prepare_data(self, batch_size: int = 2) -> Tuple[tf.Tensor, tf.Tensor]:
        self.logger.info("Data read: START")
        target_raw, context_raw = self.load_data()
        self.logger.info("Data read: END")

        self.logger.info("Convert data from file to tf.data.Dataset")
        buffer_size = len(context_raw)
        is_train = np.random.uniform(size=(len(target_raw),)) < 0.8

        train_raw = (
            tf.data.Dataset
            .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
            .shuffle(buffer_size)
            .batch(batch_size)
        )

        val_raw = (
            tf.data.Dataset
            .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
            .shuffle(buffer_size)
            .batch(batch_size)
        )
        return train_raw, val_raw
