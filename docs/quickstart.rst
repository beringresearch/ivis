.. _quickstart:

Getting Started
===============


.. code-block:: python

  from ivis import Ivis
  from sklearn import datasets

  iris = datasets.load_iris()
  X = iris.data

  # Set ivis parameters
  model = Ivis(embedding_dims=2, k=15)

  # Generate embeddings
  embeddings = model.fit_transform(X)

  # Export model
  model.save_model('iris.ivis')

  
