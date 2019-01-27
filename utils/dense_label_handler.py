"""Utilities for dense pixel label handling.
Useful for semantic segmentation purpose.
"""

import tensorflow as tf


def compute_class_weights(label, ignore_mask, class_weights=None):
    """Compute weights on loss by label data.

    Parameters
    ----------
    label: (N, H, W, C=1) tf.tensor
        Label batch used as a ground truth.
    ignore_mask: (N, H, W, C=1) tf.tensor
        A boolean mask to ignore subset of elements.
        False elements will be ignored and set as 0 in
        @p class_weights_tensor.
    class_weights: 1d tf.Tensor, default None
        Weights to validation losses over classes.
        This array will be used as the parameter of @p loss_fn.
        It should have 1d tensor with the length of the number of classes.
        If it's None, use 1 to equally weight classes.

    Returns
    -------
    class_weights_tensor: (N, H, W, C=1) tf.tensor
        Constructed weights tensor.
        This has the same shape to @p label.
    """
    if class_weights is None:
        class_weights_tensor = tf.cast(ignore_mask, tf.float32)
        return class_weights_tensor

    # Temporary set ignore id to 0 to avoid a tf.gather error on id 255.
    label_tmp = tf.multiply(label, tf.cast(ignore_mask, tf.int64))
    class_weights_tensor = tf.gather(class_weights, label_tmp)
    # Set weight 0 for ignore id.
    class_weights_tensor = tf.multiply(class_weights_tensor,
                                       tf.cast(ignore_mask, tf.float32))

    return class_weights_tensor
