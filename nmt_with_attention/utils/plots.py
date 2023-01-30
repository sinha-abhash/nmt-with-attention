from matplotlib import pyplot as plt
import tensorflow as tf


def plot_history_of_training(
    history: tf.keras.callbacks.History,
    is_plot_acc: bool = True,
    is_plot_loss: bool = True,
):
    if is_plot_acc is True:
        plot_acc(history)

    if is_plot_loss is True:
        plot_loss(history)


def plot_loss(history: tf.keras.callbacks.History):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel("Epoch #")
    plt.ylabel("CE/token")
    plt.savefig("./nmt_with_attention/plotting_images/loss.png")


def plot_acc(history: tf.keras.callbacks.History):
    plt.plot(history.history["masked_acc"], label="accuracy")
    plt.plot(history.history["val_masked_acc"], label="val_accuracy")
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel("Epoch #")
    plt.ylabel("CE/token")
    plt.savefig("./nmt_with_attention/plotting_images/acc.png")
