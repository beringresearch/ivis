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

ivis = Ivis(k=5, model='maaten', verbose=0)
ivis.fit(X)

embeddings = ivis.transform(X)

plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(embeddings[:, 0],
            embeddings[:, 1],
            c=load_iris().target, s=20)
plt.xlabel('ivis 1')
plt.ylabel('ivis 2')
plt.title('ivis embeddings of the iris dataset')

plt.show()
