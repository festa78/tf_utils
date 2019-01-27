import numpy as np
import pytest
import tensorflow as tf

from utils.dense_label_handler import compute_class_weights


class Test(tf.test.TestCase):

    def test_compute_class_weights(self):
        """Test Trainer class surely compute class weights.
        """
        with self.test_session():
            BATCH_SIZE = 2
            IMAGE_SIZE = 224
            CLASS_WEIGHTS = tf.constant(
                np.array([.1, .3, .2, .1, .3], dtype=np.float),
                dtype=tf.float32)

            label = np.ones([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
            label = tf.convert_to_tensor(label, dtype=tf.int64)
            ignore_mask = np.ones([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
            ignore_mask[1, ...] = 0
            ignore_mask = tf.convert_to_tensor(ignore_mask, dtype=tf.bool)

            class_weights_tensor = compute_class_weights(
                label, ignore_mask, CLASS_WEIGHTS)

            # The class weights for label id 1 is .3.
            self.assertAllClose(
                class_weights_tensor[0, ...],
                np.ones(class_weights_tensor[0, ...].shape) * .3)

            # The ignored part will be all zeros.
            self.assertAllClose(class_weights_tensor[1, ...],
                                np.zeros(class_weights_tensor[0, ...].shape))
