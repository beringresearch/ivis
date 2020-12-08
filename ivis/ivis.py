""" scikit-learn wrapper class for the Ivis algorithm. """
import json
import os
import shutil
import pickle as pkl
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers
from sklearn.base import BaseEstimator


from .data.generators import generator_from_neighbour_matrix, KerasSequence
from .nn.network import triplet_network, base_network
from .nn.callbacks import ModelCheckpoint
from .nn.losses import triplet_loss, is_categorical, is_multiclass, is_hinge
from .nn.losses import semi_supervised_loss, validate_sparse_labels
from .data.neighbour_retrieval import AnnoyKnnMatrix


class Ivis(BaseEstimator):
    """Ivis is a technique that uses an artificial neural network for
    dimensionality reduction, often useful for the purposes of visualization.
    The network trains on triplets of data-points at a time and pulls positive
    points together, while pushing more distant points away from each other.
    Triplets are sampled from the original data using KNN aproximation using
    the Annoy library.

    :param int embedding_dims: Number of dimensions in the embedding space
    :param int k: The number of neighbours to retrieve for each point.
        Must be less than one minus the number of rows in the dataset.
    :param Union[str,Callable] distance: The loss function used to train
        the neural network.

        *   If string: a registered loss function name. Predefined losses are:
            "pn", "euclidean", "manhattan_pn", "manhattan", "chebyshev",
            "chebyshev_pn", "softmax_ratio_pn", "softmax_ratio", "cosine", "cosine_pn".
        *   If Callable, must have two parameters, (y_true, y_pred).
            y_pred denotes the batch of triplets, and y_true are any corresponding labels.
            y_pred is expected to be of shape: (3, batch_size, embedding_dims).

                * When loading model loaded with a custom loss, provide the loss to the
                  constructor of the new Ivis instance before loading the saved model.
    :param int batch_size: The size of mini-batches used during gradient
        descent while training the neural network. Must be less than the
        num_rows in the dataset.
    :param int epochs: The maximum number of epochs to train the model for.
        Each epoch the network will see a triplet based on each data-point
        once.
    :param int n_epochs_without_progress: After n number of epochs without an
        improvement to the loss, terminate training early.
    :param int ntrees: The number of random projections trees built by Annoy to
        approximate KNN. The more trees the higher the memory usage, but the
        better the accuracy of results.
    :param int search_k: The maximum number of nodes inspected during a nearest
        neighbour query by Annoy. The higher, the more computation time
        required, but the higher the accuracy. The default is n_trees * k,
        where k is the number of neighbours to retrieve. If this is set too
        low, a variable number of neighbours may be retrieved per data-point.
    :param bool precompute: Whether to pre-compute the nearest neighbours.
        Pre-computing is a little faster, but requires more memory. If memory
        is limited, try setting this to False.
    :param Union[str,tf.keras.models.Model] model: The keras model to train using
        triplet loss.

        *   If a model object is provided, an embedding layer of size
            'embedding_dims' will be appended to the end of the network.
        *   If a string, a pre-defined network by that name will be used. Possible
            options are: 'szubert', 'hinton', 'maaten'. By default the 'szubert'
            network will be created, which is a selu network composed of 3 dense
            layers of 128 neurons each, followed by an embedding layer of size
            'embedding_dims'.
    :param str supervision_metric: The supervision metric to
        optimize when training keras in supervised mode. Supports all of the
        classification or regression losses included with keras, so long as
        the labels are provided in the correct format. A list of keras' loss
        functions can be found at https://keras.io/losses/ .
    :param float supervision_weight: Float between 0 and 1 denoting the
        weighting to give to classification vs triplet loss when training
        in supervised mode. The higher the weight, the more classification
        influences training. Ignored if using Ivis in unsupervised mode.
    :param str annoy_index_path: The filepath of a pre-trained annoy index file
        saved on disk. If provided, the annoy index file will be used.
        Otherwise, a new index will be generated and saved to disk in the
        current directory as 'annoy.index'.
    :param [keras.callbacks.Callback] callbacks: List of keras Callbacks to
        pass model during training, such as the TensorBoard callback. A set of
        ivis-specific callbacks are provided in the ivis.nn.callbacks module.
    :param bool build_index_on_disk: Whether to build the annoy index directly
        on disk. Building on disk should allow for bigger datasets to be indexed,
        but may cause issues.
    :param Union[np.array,collections.abc.Sequence] neighbour_matrix:
        Providing a neighbour matrix will cause Ivis to skip computing the Annoy KNN index
        and instead use the provided neighbour_matrix.

        - A pre-computed neighbour matrix can be provided as a numpy array. Indexing the array
          should retrieve a list of neighbours for the data point associated with that index.

        - Alternatively, dynamic computation of neighbours can be done by providing a
          class than implements the collections.abc.Sequence class, specifically the
          `__getitem__` and `__len__` methods.

            - See the ivis.data.neighbour_retrieval.AnnoyKnnMatrix class for an example.
    :param int verbose: Controls the volume of logging output the model
        produces when training. When set to 0, silences outputs, when above 0
        will print outputs.

    """

    def __init__(self, embedding_dims=2, k=150, distance='pn', batch_size=128,
                 epochs=1000, n_epochs_without_progress=20, ntrees=50,
                 knn_distance_metric='angular', search_k=-1,
                 precompute=True, model='szubert',
                 supervision_metric='sparse_categorical_crossentropy',
                 supervision_weight=0.5, annoy_index_path=None,
                 callbacks=None, build_index_on_disk=True,
                 neighbour_matrix=None, verbose=1):

        self.embedding_dims = embedding_dims
        self.k = k
        self.distance = distance
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_epochs_without_progress = n_epochs_without_progress
        self.knn_distance_metric = knn_distance_metric
        self.ntrees = ntrees
        self.search_k = search_k
        self.precompute = precompute
        self.model = model
        self.model_ = None
        self.encoder = None
        self.supervision_metric = supervision_metric
        self.supervision_weight = supervision_weight
        self.supervised_model_ = None
        self.loss_history_ = []
        self.annoy_index_path = annoy_index_path

        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback = callback.register_ivis_model(self)
        self.build_index_on_disk = build_index_on_disk
        self.neighbour_matrix = neighbour_matrix
        self.verbose = verbose
        self.n_classes = None

    def __getstate__(self):
        """ Return object serializable variable dict """

        state = dict(self.__dict__)
        if 'model_' in state:
            state['model_'] = None
        if 'encoder' in state:
            state['encoder'] = None
        if 'supervised_model_' in state:
            state['supervised_model_'] = None
        if 'callbacks' in state:
            state['callbacks'] = []
        if not isinstance(state['model'], str):
            state['model'] = None
        if 'neighbour_matrix' in state:
            state['neighbour_matrix'] = None
        if callable(state['distance']):
            state['distance'] = state['distance'].__name__
        return state

    def _fit(self, X, Y=None, shuffle_mode=True):

        if self.neighbour_matrix is None:
            if self.annoy_index_path is None:
                self.annoy_index_path = 'annoy.index'
                self.neighbour_matrix = AnnoyKnnMatrix.build(X, path=self.annoy_index_path,
                                                             k=self.k, metric=self.knn_distance_metric,
                                                             search_k=self.search_k,
                                                             include_distances=False, ntrees=self.ntrees,
                                                             build_index_on_disk=self.build_index_on_disk,
                                                             precompute=self.precompute, verbose=self.verbose)
            else:
                self.neighbour_matrix = AnnoyKnnMatrix.load(self.annoy_index_path, X.shape,
                                                            k=self.k, search_k=self.search_k,
                                                            include_distances=False, precompute=self.precompute,
                                                            verbose=self.verbose)

        datagen = generator_from_neighbour_matrix(X, Y,
                                                  neighbour_matrix=self.neighbour_matrix,
                                                  batch_size=self.batch_size)

        loss_monitor = 'loss'
        try:
            triplet_loss_func = triplet_loss(distance=self.distance)
        except KeyError:
            raise ValueError('Loss function `{}` not implemented.'.format(self.distance))

        if self.model_ is None:
            if isinstance(self.model, str):
                input_size = (X.shape[-1],)
                self.model_, anchor_embedding, _, _ = \
                    triplet_network(base_network(self.model, input_size),
                                    embedding_dims=self.embedding_dims)
            else:
                self.model_, anchor_embedding, _, _ = \
                    triplet_network(self.model,
                                    embedding_dims=self.embedding_dims)

            if Y is None:
                self.model_.compile(optimizer='adam', loss=triplet_loss_func)
            else:
                if is_categorical(self.supervision_metric):
                    if not is_multiclass(self.supervision_metric):
                        if not is_hinge(self.supervision_metric):
                            # Binary logistic classifier
                            if len(Y.shape) > 1:
                                self.n_classes = Y.shape[-1]
                            else:
                                self.n_classes = 1
                            supervised_output = Dense(self.n_classes, activation='sigmoid',
                                                      name='supervised')(anchor_embedding)
                        else:
                            # Binary Linear SVM output
                            if len(Y.shape) > 1:
                                self.n_classes = Y.shape[-1]
                            else:
                                self.n_classes = 1
                            supervised_output = Dense(self.n_classes, activation='linear',
                                                      name='supervised',
                                                      kernel_regularizer=regularizers.l2())(anchor_embedding)
                    else:
                        if not is_hinge(self.supervision_metric):
                            validate_sparse_labels(Y)
                            self.n_classes = len(np.unique(Y[Y != np.array(-1)]))
                            # Softmax classifier
                            supervised_output = Dense(self.n_classes, activation='softmax',
                                                      name='supervised')(anchor_embedding)
                        else:
                            self.n_classes = len(np.unique(Y, axis=0))
                            # Multiclass Linear SVM output
                            supervised_output = Dense(self.n_classes, activation='linear',
                                                      name='supervised',
                                                      kernel_regularizer=regularizers.l2())(anchor_embedding)
                else:
                    # Regression
                    if len(Y.shape) > 1:
                        self.n_classes = Y.shape[-1]
                    else:
                        self.n_classes = 1
                    supervised_output = Dense(self.n_classes, activation='linear',
                                              name='supervised')(anchor_embedding)

                supervised_loss = keras.losses.get(self.supervision_metric)
                if self.supervision_metric == 'sparse_categorical_crossentropy':
                    supervised_loss = semi_supervised_loss(supervised_loss)

                final_network = Model(inputs=self.model_.inputs,
                                      outputs=[self.model_.output,
                                               supervised_output])
                self.model_ = final_network
                self.model_.compile(
                    optimizer='adam',
                    loss={
                        'stacked_triplets': triplet_loss_func,
                        'supervised': supervised_loss
                    },
                    loss_weights={
                        'stacked_triplets': 1 - self.supervision_weight,
                        'supervised': self.supervision_weight})

                # Store dedicated classification model
                supervised_model_input = Input(shape=(X.shape[-1],))
                embedding = self.model_.layers[3](supervised_model_input)
                softmax_out = self.model_.layers[-1](embedding)

                self.supervised_model_ = Model(supervised_model_input, softmax_out)

        self.encoder = self.model_.layers[3]

        if self.verbose > 0:
            print('Training neural network')

        hist = self.model_.fit(
            datagen,
            epochs=self.epochs,
            callbacks=self.callbacks + [EarlyStopping(monitor=loss_monitor,
                                                      patience=self.n_epochs_without_progress)],
            shuffle=shuffle_mode,
            steps_per_epoch=int(np.ceil(X.shape[0] / self.batch_size)),
            verbose=self.verbose)
        self.loss_history_ += hist.history['loss']

    def fit(self, X, Y=None, shuffle_mode=True):
        """Fit an ivis model.

        Parameters
        ----------
        X : np.array, ivis.data.sequence.IndexableDataset, tensorflow.keras.utils.HDF5Matrix
            Data to be embedded. Needs to have a `.shape` attribute and a `__getitem__` method.
        Y : array, shape (n_samples)
            Optional array for supervised dimentionality reduction.
            If Y contains -1 labels, and 'sparse_categorical_crossentropy'
            is the loss function, semi-supervised learning will be used.

        Returns
        -------
        self: ivis.Ivis object
            Returns estimator instance.
        """

        self._fit(X, Y, shuffle_mode)
        return self

    def fit_transform(self, X, Y=None, shuffle_mode=True):
        """Fit to data then transform

        Parameters
        ----------
        X : np.array, ivis.data.sequence.IndexableDataset, tensorflow.keras.utils.HDF5Matrix
            Data to train on and then embedded.
            Needs to have a `.shape` attribute and a `__getitem__` method.
        Y : array, shape (n_samples)
            Optional array for supervised dimentionality reduction.
            If Y contains -1 labels, and 'sparse_categorical_crossentropy'
            is the loss function, semi-supervised learning will be used.

        Returns
        -------
        X_new : array, shape (n_samples, embedding_dims)
            Embedding of the data in low-dimensional space.
        """

        self.fit(X, Y, shuffle_mode)
        return self.transform(X)

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.

        Parameters
        ----------
        X : np.array, ivis.data.sequence.IndexableDataset, tensorflow.keras.utils.HDF5Matrix
            Data to be transformed. Needs to have a `.shape` attribute and a `__getitem__` method.

        Returns
        -------
        X_new : array, shape (n_samples, embedding_dims)
            Embedding of the data in low-dimensional space.
        """

        embedding = self.encoder.predict(KerasSequence(X), verbose=self.verbose)
        return embedding

    def score_samples(self, X):
        """Passes X through classification network to obtain predicted
        supervised values. Only applicable when trained in
        supervised mode.

        Parameters
        ----------
        X : np.array, ivis.data.sequence.IndexableDataset, tensorflow.keras.utils.HDF5Matrix
            Data to be passed through classification network.
            Needs to have a `.shape` attribute and a `__getitem__` method.

        Returns
        -------
        X_new : array, shape (n_samples, embedding_dims)
            Softmax class probabilities of the data.
        """

        if self.supervised_model_ is None:
            raise Exception("Model was not trained in classification mode.")

        softmax_output = self.supervised_model_.predict(X, verbose=self.verbose)
        return softmax_output

    def save_model(self, folder_path, overwrite=False):
        """Save an ivis model

        Parameters
        ----------
        folder_path : string
            Path to serialised model files and metadata
        """

        if overwrite:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
        os.makedirs(folder_path)

        # serialize weights to HDF5
        self.model_.layers[3].save(os.path.join(folder_path, 'ivis_model.h5'))
        # Have to serialize supervised model separately
        if self.supervised_model_ is not None:
            self.supervised_model_.save(os.path.join(folder_path,
                                                     'supervised_model.h5'))

        # save optimizer structure and state separately.
        # pickle preserves structure (but not correct values).
        with open(os.path.join(folder_path, 'optimizer.pkl'), 'wb') as f:
            pkl.dump(self.model_.optimizer, f)

        # save optimizer state in numpy array
        np.save(os.path.join(folder_path, 'optimizer_state.npy'),
                self.model_.optimizer.get_weights())

        json.dump(self.__getstate__(),
                  open(os.path.join(folder_path, 'ivis_params.json'), 'w'))

    def load_model(self, folder_path):
        """Load ivis model

        Parameters
        ----------
        folder_path : string
            Path to serialised model files and metadata

        Returns
        -------
        self: ivis.Ivis object
            Returns estimator instance.
        """

        ivis_config = json.load(open(os.path.join(folder_path,
                                                  'ivis_params.json'), 'r'))
        if callable(self.distance):
            ivis_config['distance'] = self.distance
        self.__dict__ = ivis_config

        loss_function = triplet_loss(self.distance)

        # ivis models trained before version 1.8.3 won't have separate optimizer file
        # maintain compatibility by falling back to old load behavior
        if not os.path.exists(os.path.join(folder_path, 'optimizer.pkl')):
            self.model_ = load_model(os.path.join(folder_path, 'ivis_model.h5'),
                                     custom_objects={'tf': tf,
                                                     loss_function.__name__: loss_function})
        else:
            base_model = load_model(os.path.join(folder_path, 'ivis_model.h5'))

            with open(os.path.join(folder_path, 'optimizer.pkl'), 'rb') as f:
                optimizer = pkl.load(f)

            optimizer_state = np.load(os.path.join(folder_path, 'optimizer_state.npy'),
                                      allow_pickle=True)
            optimizer.set_weights(optimizer_state)

            self.model_, _, _, _ = triplet_network(base_model, embedding_dims=None)
            self.model_.compile(loss=loss_function, optimizer=optimizer)

        self.encoder = self.model_.layers[3]

        # If a supervised model exists, load it
        supervised_path = os.path.join(folder_path, 'supervised_model.h5')
        if os.path.exists(supervised_path):
            self.supervised_model_ = load_model(supervised_path)
        return self
