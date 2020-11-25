from sklearn import datasets
import numpy as np
from ivis import Ivis


def test_custom_ndarray_neighbour_matrix():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    class_indicies = {label: np.argwhere(y == label).ravel() for label in np.unique(y)}
    neighbour_matrix = np.array([class_indicies[label] for label in y])

    ivis_iris = Ivis(epochs=5, neighbour_matrix=neighbour_matrix)
    ivis_iris.k = 15
    ivis_iris.batch_size = 16

    y_pred_iris = ivis_iris.fit_transform(x)
