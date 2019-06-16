[![DOI](https://zenodo.org/badge/144551119.svg)](https://zenodo.org/badge/latestdoi/144551119) [![Documentation Status](https://readthedocs.org/projects/bering-ivis/badge/?version=latest)](https://bering-ivis.readthedocs.io/en/latest/?badge=latest)

# ivis

Implementation of the ivis algorithm as described in the paper 'Structure-preserving visualisation of high dimensional single-cell datasets'.  

This algorithm uses a siamese neural network trained on triplets to reduce the dimensionality of data to two dimensions for visualization. Each triplet is sampled from one of the <i>k</i> nearest neighbours as approximated by the Annoy library, with neighbouring points being pulled together and non-neighours being pushed away.

## Installation

To install development version

```
git clone https://github.com/beringresearch/ivis
cd ivis
pip install -r requirements.txt --editable .
```

## Features
* __Scalable:__ ivis is fast and easily extends to millions of observations and thousands of features. 
* __Versatile:__ numpy arrays, sparse matrices, and hdf5 files are supported out of the box, making it easy to apply ivis to heterogeneous problems including clustering and anomaly detection.
* __Accurate:__ ivis excels at preserving both local and global features of a dataset. Often, ivis performs better at preserving global structure of the data than t-SNE, making it easy to visualise and interpret high-dimensional datasets.
* __Generalisable:__ ivis supports addition of new data points to original embeddings via a `transform` method, making it easy to incorporate ivis into standard sklearn Pipelines.

And many more! See [ivis readme](https://bering-ivis.readthedocs.io) for latest additions and examples.
 
## Examples

```
from ivis import Ivis
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data

model = Ivis(embedding_dims=2, k=15)

embeddings = model.fit_transform(X)
```

## Ivis Universe

Ivis can be used in a wide variety of real-world applications. The Ivis Universe consists of packages that extend the core Ivis functionality.

* [ivis-animate](https://github.com/beringresearch/ivis-animate) - visualise the Ivis learning process.
* [ivis-explain](https://github.com/beringresearch/ivis-explain) - explain which features contribute the most to ivis embeddings.

Copyright 2019 Bering Limited
