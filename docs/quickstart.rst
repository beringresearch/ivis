.. _quickstart:

Getting Started
===============


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

  
