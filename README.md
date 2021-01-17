[![DOI](https://joss.theoj.org/papers/10.21105/joss.01596/status.svg)](https://doi.org/10.21105/joss.01596) [![DOI](https://zenodo.org/badge/144551119.svg)](https://zenodo.org/badge/latestdoi/144551119) [![Documentation Status](https://readthedocs.org/projects/bering-ivis/badge/?version=latest)](https://bering-ivis.readthedocs.io/en/latest/?badge=latest) [![Downloads](https://pepy.tech/badge/ivis/month)](https://pepy.tech/project/ivis) [![Build Status](https://travis-ci.org/beringresearch/ivis.svg?branch=master)](https://travis-ci.org/beringresearch/ivis)

# ivis

Implementation of the ivis algorithm as described in the paper [Structure-preserving visualisation of high dimensional single-cell datasets](https://www.nature.com/articles/s41598-019-45301-0). Ivis is designed to reduce dimensionality of very large datasets using a siamese neural network trained on triplets. Both unsupervised and supervised modes are supported.

![ivis 10M data points](https://github.com/beringresearch/ivis/blob/master/docs/_static/parity_primes_ivis_1e7_16k_smaller_pts.png)


## Installation

Ivis runs on top of TensorFlow. To install the latest ivis release from PyPi running on the CPU TensorFlow package, run:

```
# TensorFlow 2 packages require a pip version >19.0.
pip install --upgrade pip
```

```
pip install ivis[cpu]
```

If you have CUDA installed and want ivis to use the tensorflow-gpu package, run

```
pip install ivis[gpu]
```

Development version can be installed directly from from github:

```
git clone https://github.com/beringresearch/ivis
cd ivis
pip install -e '.[cpu]'
```

The following **optional dependencies** are needed if using the visualization callbacks while training the Ivis model:
- matplotlib
- seaborn

## Upgrading

Ivis Python package is updated frequently! To upgrade, run:

```
pip install ivis --upgrade
```

## Features
* __Scalable:__ ivis is fast and easily extends to millions of observations and thousands of features. 
* __Versatile:__ numpy arrays, sparse matrices, and hdf5 files are supported out of the box. Additionally, both categorical and continuous features are handled well, making it easy to apply ivis to heterogeneous problems including clustering and anomaly detection.
* __Accurate:__ ivis excels at preserving both local and global features of a dataset. Often, ivis performs better at preserving global structure of the data than t-SNE, making it easy to visualise and interpret high-dimensional datasets.
* __Generalisable:__ ivis supports addition of new data points to original embeddings via a `transform` method, making it easy to incorporate ivis into standard sklearn Pipelines.

And many more! See [ivis readme](https://bering-ivis.readthedocs.io) for latest additions and examples.
 
## Examples

```
from ivis import Ivis
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
X_scaled = MinMaxScaler().fit_transform(X)

model = Ivis(embedding_dims=2, k=15)

embeddings = model.fit_transform(X_scaled)
```

Copyright 2021 Bering Limited
