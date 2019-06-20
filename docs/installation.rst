.. _installation:

Installing and getting started
==============================

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

Bugs
----

Please report any bugs you encounter through the github `issue tracker
<https://github.com/beringresearch/ivis/issues/new>`_. It will be most helpful to
include a reproducible example.
