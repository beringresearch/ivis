import pytest
import numpy as np
from ivis import Ivis


def test_iris_embedding(X, Y):
    mask = np.random.choice(range(len(Y)), size=len(Y) // 2, replace=False)
    Y[mask] = -1

    ivis_iris = Ivis(epochs=5)
    ivis_iris.k = 15
    ivis_iris.batch_size = 16

    y_pred_iris = ivis_iris.fit_transform(X, Y)

def test_correctly_indexed_semi_supervised_classificaton_classes(X, Y):
    
    # Mark points as unlabeled
    mask = np.random.choice(range(len(Y)), size=len(Y) // 2, replace=False)
    Y[mask] = -1

    supervision_metric = 'sparse_categorical_crossentropy'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    embeddings = ivis_iris.fit_transform(X, Y)

def test_non_zero_indexed_semi_supervised_classificaton_classes(X, Y):
    # Make labels non-zero indexed
    Y = Y + 1

    # Mark points as unlabeled
    mask = np.random.choice(range(len(Y)), size=len(Y) // 2, replace=False)
    Y[mask] = -1

    supervision_metric = 'sparse_categorical_crossentropy'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    with pytest.raises(ValueError):
        embeddings = ivis_iris.fit_transform(X, Y)


def test_non_consecutive_indexed_semi_supervised_classificaton_classes(X, Y):
    # Make labels non-consecutive indexed
    Y[Y == max(Y)] = max(Y) + 1

    # Mark points as unlabeled
    mask = np.random.choice(range(len(Y)), size=len(Y) // 2, replace=False)
    Y[mask] = -1

    supervision_metric = 'sparse_categorical_crossentropy'
    ivis_iris = Ivis(k=15, batch_size=16, epochs=5,
                     supervision_metric=supervision_metric)

    with pytest.raises(ValueError):
        embeddings = ivis_iris.fit_transform(X, Y)
