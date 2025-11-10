import numpy as np
import tensorflow as tf
from ivis import Ivis


def test_multidimensional_inputs():
    sample_data = np.ones(shape=(32, 8, 8, 3))

    inputs = tf.keras.layers.Input(shape=(8, 8, 3))
    x = tf.keras.layers.Conv2D(4, 3, input_shape=(8, 8, 3))(inputs)
    x =  tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    base_model = tf.keras.models.Model(inputs, x)

    model = Ivis(model=base_model, epochs=5,
                 k=4, batch_size=4)
    y_pred = model.fit_transform(sample_data)
