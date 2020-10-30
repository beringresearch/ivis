import numpy as np
import tensorflow as tf
from ivis import Ivis


def test_multidimensional_inputs():
    sample_data = np.ones(shape=(32, 8, 8, 3))

    base_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(4, 3, input_shape=(8, 8, 3)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    model = Ivis(model=base_model, epochs=5,
                 k=4, batch_size=4)
    y_pred = model.fit_transform(sample_data)
