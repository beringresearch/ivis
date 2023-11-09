.. _python_package:


Python Package
==============

Installation
------------

The latest stable release can be installed from PyPi:

.. code:: bash

  #TensorFlow 2 packages require a pip version >19.0.
  pip install --upgrade pip

.. code:: bash

  pip install ivis[cpu]

If you have CUDA installed and want ivis to use the tensorflow-gpu package, instead run ``pip install ivis[gpu]``.

.. note:: **ZSH users**. 
  If you're running ZSH, square brackets are used for globbing / pattern matching. That means `ivis` should be installed as ``pip install 'ivis[cpu]'`` or ``pip install 'ivis[gpu]'``



Alternatively, you can use ``pip`` to install the development version directly from github:

.. code-block:: bash

  pip install git+https://github.com/beringresearch/ivis.git

Another option would be to clone the github repository and install from your local copy:

.. code-block:: bash

  git clone https://github.com/beringresearch/ivis
  cd ivis
  pip install -e '.[cpu]'


Dependencies
------------

- Python 3.5+
- tensorflow
- numpy>1.14.2
- scikit-learn>0.20.0
- tqdm
- annoy


Quickstart
----------


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
