import os
import tempfile
import pytest
import pickle as pkl
import dill
from functools import partial

from sklearn import datasets
import numpy as np
import tensorflow as tf
from ivis import Ivis


@pytest.fixture(scope='function')
def model_filepath():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield os.path.join(temp_dir, 'test_ivis_model_saving_ivis')


def _validate_network_equality(model_1, model_2):
    # Serializable dict eles same
    assert model_1._get_serializable_dict() == model_2._get_serializable_dict()

    # Check all weights are the same
    for model_layer, model_2_layer in zip(model_1.encoder_.layers,
                                          model_2.encoder_.layers):
        model_layer_weights = model_layer.get_weights()
        model_2_layer_weights = model_2_layer.get_weights()
        for i in range(len(model_layer_weights)):
            assert np.all(model_layer_weights[i] == model_2_layer_weights[i])

    # Check optimizer weights are the same
    for w1, w2 in zip(model_1.model_.optimizer.get_weights(),
                      model_2.model_.optimizer.get_weights()):
        assert np.all(w1 == w2)

def _unsupervised_model_save_test(model_filepath, save_fn, load_fn):
    model = Ivis(k=15, batch_size=16, epochs=2)
    iris = datasets.load_iris()
    X = iris.data

    model.fit(X)
    save_fn(model, model_filepath)
    model_2 = load_fn(model_filepath)

    # Check that model predictions are same
    assert np.all(model.transform(X) == model_2.transform(X))
    _validate_network_equality(model, model_2)

    # Train new model
    y_pred_2 = model_2.fit_transform(X)


def _supervised_model_save_test(model_filepath, save_fn, load_fn):
    model = Ivis(k=15, batch_size=16, epochs=2,
                 supervision_metric='sparse_categorical_crossentropy')
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    model.fit(X, Y)
    save_fn(model, model_filepath)
    model_2 = load_fn(model_filepath)

    # Check that model embeddings are same
    assert np.all(model.transform(X) == model_2.transform(X))
    # Check that model supervised predictions are same
    assert np.all(model.score_samples(X) == model_2.score_samples(X))

    _validate_network_equality(model, model_2)

    # Train new model
    y_pred_2 = model_2.fit_transform(X, Y)


def _custom_model_saving(model_filepath, save_fn, load_fn):
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # Create a custom model
    inputs = tf.keras.layers.Input(shape=(X.shape[-1],))
    x = tf.keras.layers.Dense(8, activation='relu')(inputs)
    custom_model = tf.keras.Model(inputs, x)

    model = Ivis(k=15, batch_size=16, epochs=2,
                 model=custom_model)

    model.fit(X, Y)
    save_fn(model, model_filepath)
    model_2 = load_fn(model_filepath)

    # Check that model embeddings are same
    assert np.all(model.transform(X) == model_2.transform(X))
    # Check that model supervised predictions are same
    assert np.all(model.score_samples(X) == model_2.score_samples(X))

    _validate_network_equality(model, model_2)

    # Train new model
    y_pred_2 = model_2.fit_transform(X, Y)


def _supervised_custom_model_saving(model_filepath, save_fn, load_fn):
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
    save_fn(model, model_filepath)
    model_2 = load_fn(model_filepath)

    # Check that model embeddings are same
    assert np.all(model.transform(X) == model_2.transform(X))
    # Check that model supervised predictions are same
    assert np.all(model.score_samples(X) == model_2.score_samples(X))

    _validate_network_equality(model, model_2)

    # Train new model
    y_pred_2 = model_2.fit_transform(X, Y)

### Save and load ###
def _save_ivis_model(model, filepath, save_format='tf'):
    model.save_model(filepath, save_format=save_format, overwrite=True)

def _load_ivis_model(filepath):
    model_2 = Ivis()
    model_2.load_model(filepath)
    return model_2

def _pickle_ivis_model(model, filepath):
    with open(filepath, 'wb') as f:
        pkl.dump(model, f)

def _unpickle_ivis_model(filepath):
    with open(filepath, 'rb') as f:
        model = pkl.load(f)
    return model

def _dill_ivis_model(model, filepath):
    with open(filepath, 'wb') as f:
        dill.dump(model, f)

def _undill_ivis_model(filepath):
    with open(filepath, 'rb') as f:
        model = dill.load(f)
    return model

### Tests ###
### Unsupervised ###
test_ivis_model_saving = partial(_unsupervised_model_save_test,
                                 save_fn=_save_ivis_model, load_fn=_load_ivis_model)
test_ivis_model_pickling = partial(_unsupervised_model_save_test,
                                   save_fn=_pickle_ivis_model, load_fn=_unpickle_ivis_model)
test_custom_model_saving = partial(_custom_model_saving,
                                   save_fn=_save_ivis_model, load_fn=_load_ivis_model)
# dill required to serialize custom model due to non-global scope definition
test_custom_model_pickling = partial(_custom_model_saving,
                                     save_fn=_dill_ivis_model, load_fn=_undill_ivis_model)

### Supervised ###
test_supervised_model_saving = partial(_supervised_model_save_test,
                                       save_fn=_save_ivis_model, load_fn=_load_ivis_model)
test_supervised_model_pickling = partial(_supervised_model_save_test,
                                         save_fn=_pickle_ivis_model, load_fn=_unpickle_ivis_model)
test_supervised_custom_model_saving = partial(_supervised_custom_model_saving,
                                              save_fn=_save_ivis_model, load_fn=_load_ivis_model)
# dill required to serialize custom model due to non-global scope definition
test_supervised_custom_model_pickling = partial(_supervised_custom_model_saving,
                                                save_fn=_dill_ivis_model, load_fn=_undill_ivis_model)

### Other ###
test_tf_savedmodel_persistence = partial(_unsupervised_model_save_test,
                                         save_fn=partial(_save_ivis_model, save_format='tfs'),
                                         load_fn=_load_ivis_model)

def test_untrained_model_persistence(model_filepath):
    model = Ivis(k=15, batch_size=16, epochs=2)
    model.save_model(model_filepath)

def test_save_overwriting(model_filepath):
    model = Ivis(k=15, batch_size=16, epochs=2)
    iris = datasets.load_iris()
    X = iris.data

    model.fit(X)
    model.save_model(model_filepath)

    # Check that trying to save over an existing folder raises an Exception
    with pytest.raises(FileExistsError) as exception_info:
        model.save_model(model_filepath)
        assert isinstance(exception_info.value, FileExistsError)

    # Check that can overwrite existing model if requested
    model.save_model(model_filepath, overwrite=True)
