from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from ivis import Ivis

X = load_iris().data
X = MinMaxScaler().fit_transform(X)

ivis = Ivis(k=3, batch_size=120)
ivis.fit(X)
embeddings = ivis.transform(X)

ivis.save_model('iris.ivis')

