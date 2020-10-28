.. _unsupervised:

Unsupervised Dimensionality Reduction
=====================================

Dimensionality Reduction (DR) is the transformation of data from high-dimensional to low-dimensional space, whilst retaining properties of the original data in the low-dimensional space. Downstream applications range from data visualisation to machine learning and feature engineering.

Although many DR approaches exist (e.g. PCA, UMAP, t-SNE), Neural Network (NN) models have been proposed as effective non-linear alternatives. Generally, unsupervised NNs with multiple layers are trained by optimizing a target function, whilst an intermediate layer with small cardinality serves as a low dimensional representation of the input data.

We designed ``ivis`` to effectively capture local as well as global features of very large dataset. In our workflows we are applying ``ivis`` to millions of data points to effectively capture their behaviour.


The ``iris`` dataset
--------------------

To demonstrate the key features of the ``ivis`` algorithm, we will use the well-established ``iris`` dataset.

::

    from ivis import Ivis
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    data = load_iris()
    X = data.data
    y = data.target

    X = StandardScaler().fit_transform(X)

Now, let's set up ``ivis`` object.

::

    ivis = Ivis(k=15)
    ivis.fit(X)
    embeddings = ivis.transform(X)

    embeddings.shape

That's it! Note, that the ``k`` parameter is changed from the default value because we only have 150 observations in this dataset. Check out how :ref:`hyperparameters can be tuned <hyperparameters>` to get the most out of ``ivis`` for your dataset.


Reducing dimensionality of n-dimensional arrays
-----------------------------------------------

``ivis`` easily handles n-dimensional arrays. This can be useful in datasets such as imaging, where arrays are typically in (N_SAMPLES, IMG_WIDTH, IMG_HEIGHT, CHANNELS) format. To accomplish this, all we need to do is pass a custom base neural network into ivis that ensures input shapes are captured correctly.

Let's demonstrate this feature using teh ``MNSIT`` dataset.

::

    image_height, image_width = 28, 28
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], image_height, image_width, 1)
    x_test = x_test.reshape(x_test.shape[0], image_height, image_width, 1)
    input_shape = (image_height, image_width, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255


We now define the custom neural network that will be used as a feature extractor. Since we are dealing with images, we can use convolutional blocks:

::

    def get_base_network(in_shape):
        inputs = tf.keras.layers.Input(in_shape)
        x = tf.keras.layers.Convolution2D(32, (3,3), activation='relu', kernel_initializer='he_uniform')(inputs)
        x = tf.keras.layers.MaxPool2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        model = tf.keras.models.Model(inputs, x)
        return model

    in_shape = x_train.shape[1:]
    base_model = maaten_base_network(in_shape)

Once the network is set up, all we have to do is let ``Ivis`` know that we will be using a custom network rather than the pre-built one.

::

    ivis = Ivis(model=base_model)
    ivis.fit(x_train)

    embeddings = ivis.transform(x_train)
    embeddings.shape

All done - you have just reduced dimensionality of an imaging dataset!


If you're looking to extract the finetuned base model from the ivis triplet loss network, you can grab it directlu from the ``ivis`` object:

::

    model = ivis.model_.layers[3]


Using custom KNN retreaval
--------------------------

``ivis`` uses Annoy to retreave nearest neighbours during tripplet selection. Annoy was selected as the default option because its fast, accurate and a nearest neighbour index can be built on directly disk, meaning that massive datasets can be processed without the need to load them into memory.

However, many other algorithms exist and new ones are popping up continuously. To accommodate custom nearest neighbour selection, ``ivis`` can accept a nearest neighbour matrix directly through the ``neighbour_matrix`` hyperparameter.

::

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=15).fit(X)
    neighbours = nn.kneighbors(X, return_distance=False) 

    ivis = Ivis(neighbour_matrix=neighbours)
    ivis.fit(X)