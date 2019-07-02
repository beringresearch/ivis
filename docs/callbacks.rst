.. _callbacks:

Callbacks
==========

Callbacks are called periodically during training of the ``ivis`` model. 
These allow you to get an insight into the progress being made during 
training. With this information, you may decide to terminate a training 
session early due to a lack of improvements to the visualizations, for 
example. They also provide helpful logging features, allowing you to periodically  
save a checkpoint of an ``ivis`` model which can be used to resuming training later.

To use a callback during training, simply pass a list of callback objects to 
the ``Ivis`` object when creating it using the ``callbacks`` keyword argument.
The ``ivis.nn.callbacks`` module contains a set of callbacks provided for 
use with ``ivis`` models. However, any ``keras.callbacks.Callbacks`` object can be 
passed and will be used during training: for example, ``keras.callbacks.TensorBoard``.


ModelCheckpoint
-------------------

.. currentmodule:: ivis.nn.callbacks

.. autoclass:: ModelCheckpoint
  :undoc-members:
  :show-inheritance:

EmbeddingsLogging
---------------------

.. currentmodule:: ivis.nn.callbacks

.. autoclass:: EmbeddingsLogging
  :undoc-members:
  :show-inheritance:

EmbeddingsImage
----------------

.. currentmodule:: ivis.nn.callbacks

.. autoclass:: EmbeddingsImage
  :undoc-members:
  :show-inheritance:

TensorBoardEmbeddingsImage
---------------------------

.. currentmodule:: ivis.nn.callbacks

.. autoclass:: TensorBoardEmbeddingsImage
  :undoc-members:
  :show-inheritance:

.. raw:: html

    <video controls loop="true" autoplay="autoplay" width="560" height="315" src="_static/tensorboard_embeddings_plots.mp4"></video>
