import tensorflow as tf

from nmt_with_attention.data import Preprocessor, DataReader


def test_vectorization():
    dr = DataReader(dataset_path="./test_spa_eng.txt")
    train, val = dr.prepare_data(batch_size=2)
    preprocessor = Preprocessor(train=train, val=val)
    preprocessor.vectorization()
    assert isinstance(preprocessor.context_text_preprocessor.get_vocabulary(), list)
    assert isinstance(preprocessor.target_text_preprocessor.get_vocabulary(), list)


def test_process_text():
    dr = DataReader(dataset_path="./test_spa_eng.txt")
    train, val = dr.prepare_data(batch_size=2)
    preprocessor = Preprocessor(train=train, val=val)
    preprocessor.vectorization()

    train_ds = train.map(preprocessor.process_text, tf.data.AUTOTUNE)
    assert len(list(train_ds.as_numpy_iterator())) != 0
