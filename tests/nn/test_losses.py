import tempfile
import os
import pytest
from tensorflow.keras import backend as K
import tensorflow as tf

from ivis.nn import losses as losses
from ivis.nn.distances import euclidean_distance
from ivis import Ivis


@pytest.fixture(scope='function')
def model_filepath():
    with tempfile.TemporaryDirectory() as temp_dir:
        fpath = os.path.join(temp_dir, 'test_loss_plugin.ivis')
        yield fpath


def test_loss_function_call():
    for loss_name in losses.loss_dict:
        # Attempt to construct loss by name
        losses.triplet_loss(distance=loss_name)

def test_custom_loss_fn_registration():
    @losses.register_loss
    def custom_loss_fn(y_true, y_pred):
        return y_pred - y_true
    assert custom_loss_fn.__name__ in losses.loss_dict
    assert custom_loss_fn is losses.loss_dict[custom_loss_fn.__name__]
    assert losses.triplet_loss(distance=custom_loss_fn.__name__) is custom_loss_fn

def test_custom_loss_ivis(X, model_filepath):

    def euclidean_loss(y_true, y_pred):
        margin = 1
        anchor, positive, negative = tf.unstack(y_pred)
        return K.mean(K.maximum(euclidean_distance(anchor, positive) - euclidean_distance(anchor, negative) + margin, 0))

    model = Ivis(distance=euclidean_loss, k=15, batch_size=16, epochs=3)
    y_pred = model.fit_transform(X)

    # Test model saving and loading
    model.save_model(model_filepath, overwrite=True)
    model_2 = Ivis(distance=euclidean_loss)
    model_2.load_model(model_filepath)

    model_3 = Ivis()
    with pytest.raises(ValueError):
        model_3.load_model(model_filepath)

def test_custom_loss_ivis_callable(X, model_filepath):

    class EuclideanDistance:
        def __init__(self, margin=1):
            self.margin = margin
            self.__name__ = self.__class__.__name__
        def _euclidean_distance(self, x, y):
            return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=-1, keepdims=True), K.epsilon()))
        def __call__(self, y_true, y_pred):
            anchor, positive, negative = tf.unstack(y_pred)
            return K.mean(K.maximum(self._euclidean_distance(anchor, positive) - self._euclidean_distance(anchor, negative) + self.margin, 0))

    model = Ivis(distance=EuclideanDistance(margin=2), k=15, batch_size=16, epochs=5)
    y_pred = model.fit_transform(X)

    # Test model saving and loading
    model.save_model(model_filepath, overwrite=True)
    model_2 = Ivis(distance=EuclideanDistance(margin=2))
    model_2.load_model(model_filepath)
    model_2.fit(X)
