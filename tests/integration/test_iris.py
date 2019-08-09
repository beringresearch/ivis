from ivis import Ivis
from sklearn import datasets
import numpy as np


def test_iris_embedding():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    ivis_iris = Ivis(n_epochs_without_progress=5)
    ivis_iris.k = 15
    ivis_iris.batch_size = 16

    y_pred_iris = ivis_iris.fit_transform(x)
