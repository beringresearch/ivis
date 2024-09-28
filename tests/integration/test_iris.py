from ivis import Ivis


def test_iris_embedding(X):
    ivis_iris = Ivis(epochs=5)
    ivis_iris.k = 15
    ivis_iris.batch_size = 16

    y_pred_iris = ivis_iris.fit_transform(X)

def test_1d_iris_embedding(X):
    ivis_iris = Ivis(epochs=5, embedding_dims=1)
    ivis_iris.k = 15
    ivis_iris.batch_size = 16

    y_pred_iris = ivis_iris.fit_transform(X)
