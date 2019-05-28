""" ivis clustering"""

import hdbscan
import matplotlib
matplotlib.use('Agg')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from ivis import Ivis

sns.set(style='white', rc={'figure.figsize': (22, 7)})

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Generate ivis embeddings
ivis = Ivis(k=250, model='maaten')
ivis.fit(X)
ivis.save_model('ivis.mnist')

ivis = Ivis(k=250)
ivis.load_model('ivis.mnist')

embedding = ivis.transform(X)

# Kmeans
kmeans = KMeans(n_clusters=10).fit(X)

# HDBSCAN
labels = hdbscan.HDBSCAN(
            min_samples=1,
            min_cluster_size=500).fit_predict(embedding)

coordinates = []
for l in range(10):
    coordinates.append((np.average(embedding[np.where(y == str(l)),  0]),
                        np.average(embedding[np.where(y == str(l)), 1])))
    cluster_labels = {key: value for (key, value) in zip(list(range(10)),
                                                         coordinates)}

plt.cla()
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.scatter(embedding[:, 0],
            embedding[:, 1], c=[int(l) for l in y], s=0.1, cmap='Spectral')
for l in range(10):
    ax1.annotate(l, cluster_labels[l])
ax1.title.set_text('Original Labels')
ax2.scatter(embedding[:, 0],
            embedding[:, 1], c=kmeans.labels_, s=0.1, cmap='Spectral')
ax2.title.set_text('K-means (10 clusters)')
ax3.scatter(embedding[:, 0],
            embedding[:, 1], c=labels, s=0.1, cmap='Spectral')
ax3.title.set_text('HBSCAN')
plt.savefig('mnist.png', bbox_inches='tight')


adjusted_rand_score([int(l) for l in y], labels)
