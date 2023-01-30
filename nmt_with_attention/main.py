from argparse import ArgumentParser
import logging
import tensorflow as tf

from nmt_with_attention.data import DataReader, Preprocessor
from nmt_with_attention import config
from nmt_with_attention.model import Translator, Export
from nmt_with_attention.utils import masked_acc, masked_loss, plot_history_of_training


logging.basicConfig(level=logging.INFO)


def serve(args):
    logger = logging.getLogger("serve")
    dr = DataReader(args.dataset_path)
    train, val = dr.prepare_data(batch_size=config.BATCH_SIZE)
    logger.info("data reading completed")

    preprocessor = Preprocessor(train=train, val=val)
    train_ds = train.map(preprocessor.process_text, tf.data.AUTOTUNE)
    val_ds = val.map(preprocessor.process_text, tf.data.AUTOTUNE)
    logger.info("data preprocessing completed")

    logger.info("create model")
    model = Translator(
        config.UNITS,
        preprocessor.context_text_preprocessor,
        preprocessor.target_text_preprocessor,
    )

    logger.info("compile model")
    model.compile(optimizer="adam", loss=masked_loss, metrics=[masked_acc, masked_loss])

    logger.info("train the model")
    history: tf.keras.callbacks.History = model.fit(
        train_ds.repeat(),
        epochs=100,
        steps_per_epoch=100,
        validation_data=val_ds,
        validation_steps=20,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
    )
    logger.info("model trained")
    plot_history_of_training(history=history)
    logger.info("plots saved")

    logger.info("exporting model")
    export = Export(model=model)
    tf.saved_model.save(
        export,
        "./nmt_with_attention/trained_models/translator",
        signatures={"serving_default": export.translate},
    )
    logger.info("model exported!")


def main():
    parser = ArgumentParser(description="Practice Project for NMT")
    parser.add_argument("--dataset_path", "-d", type=str, help="Provide dataset path")
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
