from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from ivis import Ivis

X = load_iris().data
X = MinMaxScaler().fit_transform(X)

embeddings = Ivis(k=3, batch_size=120).fit_transform(X)


