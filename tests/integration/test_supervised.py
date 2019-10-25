from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses
from ivis import Ivis
import numpy as np
import pytest
from sklearn import datasets


def test_score_samples_unsupervised():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    ivis_iris = Ivis(k=15, batch_size=16, epochs=5)
    embeddings = ivis_iris.fit_transform(x)

    # Unsupervised model cannot classify
    with pytest.raises(Exception):
        y_pred = ivis_iris.score_samples(x)


def test_score_samples():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    supervision_metric = 'sparse_categorical_crossentropy'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    embeddings = ivis_iris.fit_transform(x, y)
    y_pred = ivis_iris.score_samples(x)

    # Softmax probabilities add to one, correct shape
    assert np.sum(y_pred, axis=-1) == pytest.approx(1, 0.01)
    assert y_pred.shape[0] == x.shape[0]
    assert y_pred.shape[1] == len(np.unique(y))

    # Check that loss function and activation are correct
    loss_name = ivis_iris.model_.loss['supervised'].__name__
    assert losses.get(loss_name).__name__  == losses.get(supervision_metric).__name__
    assert ivis_iris.model_.layers[-1].activation.__name__ == 'softmax'


def test_correctly_indexed_classificaton_classes():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    supervision_metric = 'sparse_categorical_crossentropy'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    embeddings = ivis_iris.fit_transform(x, y)


def test_non_zero_indexed_classificaton_classes():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Make labels non-zero indexed
    y = y + 1

    supervision_metric = 'sparse_categorical_crossentropy'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    with pytest.raises(ValueError):
        embeddings = ivis_iris.fit_transform(x, y)


def test_non_consecutive_indexed_classificaton_classes():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Make labels non-consecutive indexed
    y[y == max(y)] = max(y) + 1

    supervision_metric = 'sparse_categorical_crossentropy'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    with pytest.raises(ValueError):
        embeddings = ivis_iris.fit_transform(x, y)


def test_invalid_metric():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    supervision_metric = 'invalid_loss_function'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    # Loss function not specified
    with pytest.raises(ValueError):
        embeddings = ivis_iris.fit_transform(x, y)


def test_svm_score_samples():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    supervision_metric = 'categorical_hinge'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    # Correctly formatted one-hot labels train successfully
    y = to_categorical(y)
    embeddings = ivis_iris.fit_transform(x, y)

    y_pred = ivis_iris.score_samples(x)

    loss_name = ivis_iris.model_.loss['supervised'].__name__
    assert losses.get(loss_name).__name__ == losses.get(supervision_metric).__name__
    assert ivis_iris.model_.layers[-1].activation.__name__ == 'linear'
    assert ivis_iris.model_.layers[-1].kernel_regularizer is not None
    assert ivis_iris.model_.layers[-1].output_shape[-1] == y.shape[-1]


def test_regression():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    supervision_metric = 'mae'
    ivis_boston = Ivis(k=15, batch_size=16, epochs=5,
                       supervision_metric=supervision_metric)
    ivis_boston.fit(x_train, y_train)

    embeddings = ivis_boston.transform(x_train)
    y_pred = ivis_boston.score_samples(x_train)

    loss_name = ivis_boston.model_.loss['supervised'].__name__
    assert losses.get(loss_name).__name__ == losses.get(supervision_metric).__name__
    assert ivis_boston.model_.layers[-1].activation.__name__ == 'linear'
    assert ivis_boston.model_.layers[-1].output_shape[-1] == 1
