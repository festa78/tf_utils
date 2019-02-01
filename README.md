# tf_utils
TensorFlow utilities.

* `image_and_label_processing.py`  
Utils to process image and label pair.

* `tfrecord_dense_label.py`  
tfrecord read/write functions for dense pixel label, e.g. semantic segmentation.

* `tfrecord_box_label.py`  
tfrecord read/write functions for box label, e.g. object detection.

* `dense_label_handler.py`  
Utilities for dense pixel label handling.
Useful for semantic segmentation purpose.

* `box_label_handler.py`  
Utilities for bounding box label handling.
Useful for object detection purpose.

* `common_layers.py`  
Useful common functionals for neural network definitions.

* `data_preprocessor.py`  
Process tf.data.Dataset to add various pre-processing on the top.
