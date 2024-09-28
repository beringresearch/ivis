import numpy as np
from ivis import Ivis


def test_custom_ndarray_neighbour_matrix(X, Y):

    class_indicies = {label: np.argwhere(Y == label).ravel() for label in np.unique(Y)}
    neighbour_matrix = np.array([class_indicies[label] for label in Y])

    ivis_iris = Ivis(epochs=5, neighbour_matrix=neighbour_matrix)
    ivis_iris.k = 15
    ivis_iris.batch_size = 16

    y_pred_iris = ivis_iris.fit_transform(X)
