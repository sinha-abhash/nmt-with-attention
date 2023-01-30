from argparse import ArgumentParser
import logging
from pathlib import Path

import tensorflow as tf

from nmt_with_attention.model import Translator


logging.basicConfig(level=logging.INFO)


class NotCorrectModelPathException(Exception):
    pass


def check_if_model_exists(path: Path) -> bool:
    if not path.exists:
        raise FileNotFoundError(f"Provided path does not exists: {str(path)}")

    all_files = list(path.iterdir())
    model_files = [f for f in all_files if f.suffix == ".pb"]

    if len(model_files) == 0:
        raise NotCorrectModelPathException(f"Provided path does not have model files.")

    return True


def inference(args):
    logger = logging.getLogger("inference")
    check_if_model_exists(Path(args.model_path))
    logger.info("Loading model from %s", args.model_path)
    model: Translator = tf.saved_model.load(args.model_path)
    result = model.translate(tf.constant([args.input]))
    logger.info("Decoded result: %s", result[0].numpy().decode())


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", type=str, help="Input string to try the model")
    parser.add_argument("--model_path", "-m", type=str, help="Model path")
    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
