from ivis import Ivis
from sklearn import datasets
import numpy as np
import pytest


def test_classification_proba_unsupervised():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    ivis_iris = Ivis(k=15, batch_size=16, epochs=5)
    embeddings = ivis_iris.fit_transform(x)

    # Unsupervised model cannot classify
    with pytest.raises(Exception):
        y_pred = ivis_iris.classification_proba(x)

def test_classification_proba():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    ivis_iris = Ivis(k=15, batch_size=16, epochs=5)

    embeddings = ivis_iris.fit_transform(x, y)
    y_pred = ivis_iris.classification_proba(x)

    # Softmax probabilities add to one, correct shape
    assert np.sum(y_pred, axis=-1) == pytest.approx(1, 0.01)
    assert y_pred.shape[0] == x.shape[0]
    assert y_pred.shape[1] == len(np.unique(y))