import pytest
import numpy as np
from nmt_with_attention.data import DataReader

import tensorflow as tf


def test_initialization():
    with pytest.raises(FileNotFoundError):
        DataReader("wrong_path.text")


def test_load_data():
    dr = DataReader(dataset_path="./test_spa_eng.txt")
    target, context = dr.load_data()
    assert isinstance(target, np.ndarray)
    assert isinstance(context, np.ndarray)


def test_prepare_data():
    batch_size = 2
    dr = DataReader(dataset_path="./test_spa_eng.txt")
    train, val = dr.prepare_data(batch_size=batch_size)

    assert train is not None
    assert val is not None
    assert isinstance(train, tf.data.Dataset)
    assert isinstance(val, tf.data.Dataset)

    assert len(list(train.as_numpy_iterator())) == batch_size
    assert len(list(val.as_numpy_iterator())) == batch_size
