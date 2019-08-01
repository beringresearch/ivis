.. _python_package:


Python Package
==============

Installation
------------

The latest stable release can be installed from PyPi:

.. code:: bash

  pip install ivis


Alternatively, you can use ``pip`` to install the development version directly from github:

.. code-block:: bash

  pip install git+https://github.com/beringresearch/ivis.git

Another option would be to clone the github repository and install from your local copy:

.. code-block:: bash

  git clone https://github.com/beringresearch/ivis
  cd ivis
  pip install -r requirements.txt -e .


Dependencies
------------

- Python 3.5+
- tensorflow
- keras
- numpy>1.14.2
- scikit-learn>0.20.0
- tqdm
- annoy

.. code-block:: python

  from ivis import Ivis
  from sklearn.preprocessing import MinMaxScaler
  from sklearn import datasets

  iris = datasets.load_iris()
  X = iris.data

  # Scale the data to [0, 1]
  X_scaled = MinMaxScaler().fit_transform(X)

  # Set ivis parameters
  model = Ivis(embedding_dims=2, k=15)

  # Generate embeddings
  embeddings = model.fit_transform(X_scaled)

  # Export model
  model.save_model('iris.ivis')


Getting Started
---------------


.. code-block:: python

  from ivis import Ivis
  from sklearn.preprocessing import MinMaxScaler
  from sklearn import datasets

  iris = datasets.load_iris()
  X = iris.data

  # Scale the data to [0, 1]
  X_scaled = MinMaxScaler().fit_transform(X)

  # Set ivis parameters
  model = Ivis(embedding_dims=2, k=15)

  # Generate embeddings
  embeddings = model.fit_transform(X_scaled)

  # Export model
  model.save_model('iris.ivis')


Bugs
----

Please report any bugs you encounter through the github `issue tracker
<https://github.com/beringresearch/ivis/issues/new>`_. It will be most helpful to
include a reproducible example.
