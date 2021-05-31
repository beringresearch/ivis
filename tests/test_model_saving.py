import os
import tempfile
import pytest
from sklearn import datasets
import numpy as np
import tensorflow as tf
from ivis import Ivis


@pytest.fixture(scope='function')
def model_filepath():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield os.path.join(temp_dir, 'test.ivis.model.saving.ivis')


def test_ivis_model_saving(model_filepath):
    model = Ivis(k=15, batch_size=16, epochs=2)
    iris = datasets.load_iris()
    X = iris.data

    model.fit(X)
    model.save_model(model_filepath, overwrite=True)

    model_2 = Ivis()
    model_2.load_model(model_filepath)

    # Check that model predictions are same
    assert np.all(model.transform(X) == model_2.transform(X))
    # Serializable dict eles same
    assert model.__getstate__() == model_2.__getstate__()

    # Check all weights are the same
    for model_layer, model_2_layer in zip(model.encoder.layers,
                                          model_2.encoder.layers):
        model_layer_weights = model_layer.get_weights()
        model_2_layer_weights = model_2_layer.get_weights()
        for i in range(len(model_layer_weights)):
            assert np.all(model_layer_weights[i] == model_2_layer_weights[i])

    # Check optimizer weights are the same
    for w1, w2 in zip(model.model_.optimizer.get_weights(),
                      model_2.model_.optimizer.get_weights()):
        assert np.all(w1 == w2)

    # Check that trying to save over an existing folder raises an Exception
    with pytest.raises(FileExistsError) as exception_info:
        model.save_model(model_filepath)
        assert isinstance(exception_info.value, FileExistsError)

    # Check that can overwrite existing model if requested
    model.save_model(model_filepath, overwrite=True)

    # Train new model
    y_pred_2 = model_2.fit_transform(X)

def test_supervised_model_saving(model_filepath):
    model = Ivis(k=15, batch_size=16, epochs=2,
                 supervision_metric='sparse_categorical_crossentropy')
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    model.fit(X, Y)
    model.save_model(model_filepath, overwrite=True)

    model_2 = Ivis()
    model_2.load_model(model_filepath)

    # Check that model embeddings are same
    assert np.all(model.transform(X) == model_2.transform(X))
    # Check that model supervised predictions are same
    assert np.all(model.score_samples(X) == model_2.score_samples(X))
    # Serializable dict eles same
    assert model.__getstate__() == model_2.__getstate__()

    # Check all weights are the same
    for model_layer, model_2_layer in zip(model.encoder.layers,
                                          model_2.encoder.layers):
        model_layer_weights = model_layer.get_weights()
        model_2_layer_weights = model_2_layer.get_weights()
        for i in range(len(model_layer_weights)):
            assert np.all(model_layer_weights[i] == model_2_layer_weights[i])

    # Check optimizer weights are the same
    for w1, w2 in zip(model.model_.optimizer.get_weights(),
                      model_2.model_.optimizer.get_weights()):
        assert np.all(w1 == w2)

    # Check that trying to save over an existing folder raises an Exception
    with pytest.raises(FileExistsError) as exception_info:
        model.save_model(model_filepath)
        assert isinstance(exception_info.value, FileExistsError)

    # Check that can overwrite existing model if requested
    model.save_model(model_filepath, overwrite=True)

    # Train new model
    y_pred_2 = model_2.fit_transform(X, Y)

def test_custom_model_saving(model_filepath):
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # Create a custom model
    inputs = tf.keras.layers.Input(shape=(X.shape[-1],))
    x = tf.keras.layers.Dense(8, activation='relu')(inputs)
    custom_model = tf.keras.Model(inputs, x)

    model = Ivis(k=15, batch_size=16, epochs=2,
                 supervision_metric='sparse_categorical_crossentropy',
                 model=custom_model)

    model.fit(X, Y)
    model.save_model(model_filepath, overwrite=True)

    model_2 = Ivis()
    model_2.load_model(model_filepath)

    # Check that model embeddings are same
    assert np.all(model.transform(X) == model_2.transform(X))
    # Check that model supervised predictions are same
    assert np.all(model.score_samples(X) == model_2.score_samples(X))
    # Serializable dict eles same
    assert model.__getstate__() == model_2.__getstate__()

    # Check all weights are the same
    for model_layer, model_2_layer in zip(model.encoder.layers,
                                          model_2.encoder.layers):
        model_layer_weights = model_layer.get_weights()
        model_2_layer_weights = model_2_layer.get_weights()
        for i in range(len(model_layer_weights)):
            assert np.all(model_layer_weights[i] == model_2_layer_weights[i])

    # Check optimizer weights are the same
    for w1, w2 in zip(model.model_.optimizer.get_weights(),
                      model_2.model_.optimizer.get_weights()):
        assert np.all(w1 == w2)

    # Train new model
    y_pred_2 = model_2.fit_transform(X, Y)
