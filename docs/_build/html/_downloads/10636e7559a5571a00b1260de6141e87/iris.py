"""
iris dataset
============

Example of reducing dimensionality of the iris dataset using ivis.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

from ivis import Ivis

sns.set(context='paper', style='white')

X = load_iris().data
X = MinMaxScaler().fit_transform(X)

ivis = Ivis(k=3, batch_size=120, model='maaten')
ivis.fit(X)

embeddings = ivis.transform(X)

fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(
        embeddings[:, 0], embeddings[:, 1], c=load_iris().target, cmap='Spectral', s=1
            )
plt.setp(ax, xticks=[], yticks=[])
plt.title('ivis embeddings of the iris dataset', fontsize=18)

plt.show()
