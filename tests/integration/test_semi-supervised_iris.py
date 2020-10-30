import pytest
from sklearn import datasets
import numpy as np
from ivis import Ivis


def test_iris_embedding():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    mask = np.random.choice(range(len(y)), size=len(y) // 2, replace=False)
    y[mask] = -1

    ivis_iris = Ivis(epochs=5)
    ivis_iris.k = 15
    ivis_iris.batch_size = 16

    y_pred_iris = ivis_iris.fit_transform(x, y)

def test_correctly_indexed_semi_supervised_classificaton_classes():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    
    # Mark points as unlabeled
    mask = np.random.choice(range(len(y)), size=len(y) // 2, replace=False)
    y[mask] = -1

    supervision_metric = 'sparse_categorical_crossentropy'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    embeddings = ivis_iris.fit_transform(x, y)

def test_non_zero_indexed_semi_supervised_classificaton_classes():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Make labels non-zero indexed
    y = y + 1

    # Mark points as unlabeled
    mask = np.random.choice(range(len(y)), size=len(y) // 2, replace=False)
    y[mask] = -1

    supervision_metric = 'sparse_categorical_crossentropy'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    with pytest.raises(ValueError):
        embeddings = ivis_iris.fit_transform(x, y)


def test_non_consecutive_indexed_semi_supervised_classificaton_classes():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    # Make labels non-consecutive indexed
    y[y == max(y)] = max(y) + 1

    # Mark points as unlabeled
    mask = np.random.choice(range(len(y)), size=len(y) // 2, replace=False)
    y[mask] = -1

    supervision_metric = 'sparse_categorical_crossentropy'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    with pytest.raises(ValueError):
        embeddings = ivis_iris.fit_transform(x, y)
