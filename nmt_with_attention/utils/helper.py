import tensorflow as tf
import tensorflow_text as tf_text


def tf_lower_and_split_punct(text: str) -> tf.Tensor:
    text = tf_text.normalize_utf8(text, "NFKD")
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, "[^ a-z.?!,¿]", "")
    text = tf.strings.regex_replace(text, "[.?!,¿]", r" \0 ")
    text = tf.strings.strip(text)

    return tf.strings.join(["[START]", text, "[END]"], separator=" ")


def masked_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match) / tf.reduce_sum(mask)
