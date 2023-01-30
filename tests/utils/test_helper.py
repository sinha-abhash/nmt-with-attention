import numpy as np
from nmt_with_attention.utils import tf_lower_and_split_punct


def test_tf_lower_and_split_punct():
    test_string = "¿Todavía está en casa?"
    result = tf_lower_and_split_punct(test_string)
    assert result is not None
    assert np.char.decode(result.numpy()) == "[START] ¿ todavia esta en casa ? [END]"
