ivis dimensionality reduction
=============================

|fig1| |fig2|

.. |fig1| image:: _static/ivis_aorta_all_markers.png
  :width: 49 %

.. |fig2| image:: _static/ivis_retinal_bipolar_cells.png
  :width: 49 %

``ivis`` is a machine learning library for reducing dimensionality of very large datasets using Siamese Neural Networks. ``ivis`` preserves global data structures in a low-dimensional space, adds new data points to existing embeddings using a parametric mapping function, and scales linearly to millions of observations. The algorithm is described in detail in `Structure-preserving visualisation of high dimensional single-cell datasets <https://www.nature.com/articles/s41598-019-45301-0>`_.


Features
--------

* Unsupervised, semi-supervised, and fully supervised dimensionality reduction

* Support for arbitrary datasets
   
   * N-dimensional numpy arrays
   * Sparse matrices
   * Image files on disk
   * Custom data connectors

* In- and out-of-memory data processing
* Arbitrary neural network backbones 
* Callbacks and Tensorboard integration



The latest development version is on `github <https://github.com/beringresearch/ivis>`_.

.. toctree::
   :maxdepth: 2
   :caption: Get Started
  
   Python Package <python_package>
   R Package <r_package>

.. toctree::
   :maxdepth: 2
   :caption: Using ivis

   Unsupervised Dimensionality Reduction <unsupervised>
   Supervised Dimensionality Reduction <supervised>
   Semi-supervised Dimensionality Reduction <semi_supervised>
   Hyperparameter Selection <hyperparameters>
   Callbacks <callbacks>
   Examples <auto_examples/index>

.. toctree::
   :maxdepth: 2
   :caption: Applications
  
   Visualising Single Cell Experiments <scanpy_singlecell> 
   Dimensionality Reduction <comparisons>
   Metric Learning <metric_learning>
   Out-of-memory Datasets <oom_datasets>

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   Speed of Execution <timings_benchmarks>
   Distance Preservation <embeddings_benchmarks>

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   API Guide <api>
