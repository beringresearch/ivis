""" scikit-learn wrapper class for the Ivis algorithm. """
from .data.triplet_generators import create_triplet_generator_from_index_path
from .nn.network import build_network, base_network
from .nn.losses import triplet_loss
from .data.knn import build_annoy_index

from keras.callbacks import EarlyStopping
from keras.models import model_from_json, load_model

from sklearn.base import BaseEstimator

import json
import os
import multiprocessing
import tensorflow as tf

class Ivis(BaseEstimator):
    """Ivis is a technique that uses an artificial neural network for dimensionality reduction, often useful for the purposes of visualization.  
    The network trains on triplets of data-points at a time and pulls positive points together, while pushing more distant points away from each other.  
    Triplets are sampled from the original data using KNN aproximation using the Annoy library.
    
    :param int embedding_dims: Number of dimensions in the embedding space
    :param int k: The number of neighbours to retrieve for each point. Must be less than one minus the number of rows in the dataset.
    :param str distance: The loss function used to train the neural network. One of "pn", "euclidean", "manhattan_pn", "manhattan", "chebyshev", "chebyshev_pn", "softmax_ratio_pn", "softmax_ratio".
    :param int batch_size: The size of mini-batches used during gradient descent while training the neural network. Must be less than the num_rows in the dataset.
    :param int epochs: The maximum number of epochs to train the model for. Each epoch the network will see a triplet based on each data-point once.
    :param int n_epochs_without_progress: After n number of epochs without an improvement to the loss, terminate training early.
    :param float margin: The distance that is enforced between points by the triplet loss functions.
    :param int ntrees: The number of random projections trees built by Annoy to approximate KNN. The more trees the higher the memory usage, but the better the accuracy of results.
    :param int search_k: The maximum number of nodes inspected during a nearest neighbour query by Annoy. The higher, the more computation time required, but the higher the accuracy. The default is n_trees * k, where k is the number of neighbours to retrieve. If this is set too low, a variable number of neighbours may be retrieved per data-point.
    :param bool precompute: Whether to pre-compute the nearest neighbours. Pre-computing is significantly faster, but requires more memory. If memory is limited, try setting this to False.
    :param str model: str or keras.models.Model. The keras model to train using triplet loss. If a model object is provided, an embedding layer of size 'embedding_dims' will be appended to the end of the network. If a string, a pre-defined network by that name will be used. Possible options are: 'default', 'hinton', 'maaten'. By default, a selu network composed of 3 dense layers of 128 neurons each will be created, followed by an embedding layer of size 'embedding_dims'.
    :param str annoy_index_path: The filepath of a pre-trained annoy index file saved on disk. If provided, the annoy index file will be used. Otherwise, a new index will be generated and saved to disk in the current directory as 'annoy.index'.
    :param int verbose: Controls the volume of logging output the model produces when training. When set to 0, silences outputs, when above 0 will print outputs.

    """
    
    def __init__(self, embedding_dims=2, k=150, distance='pn', batch_size=128,
                        epochs=1000, n_epochs_without_progress=50,
                        margin=1, ntrees=50, search_k=-1,
                        precompute=True, model='default', annoy_index_path=None, verbose=1):

        self.embedding_dims = embedding_dims
        self.k = k
        self.distance = distance
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_epochs_without_progress = n_epochs_without_progress
        self.margin = margin
        self.ntrees = ntrees
        self.search_k = search_k
        self.precompute = precompute
        self.model_def = model
        self.model_ = None
        self.encoder = None
        self.loss_history_ = []
        self.annoy_index_path = annoy_index_path
        self.verbose = verbose

    def __getstate__(self):
        """ Return object serializable variable dict """

        state = dict(self.__dict__)
        del state['model_']
        del state['encoder']
        return state

    def _fit(self, X, shuffle_mode=True):
        
        if self.annoy_index_path is None:
            self.annoy_index_path = 'annoy.index'
            if self.verbose > 0:
                print('Building KNN index')
            build_annoy_index(X, self.annoy_index_path, ntrees=self.ntrees, verbose=self.verbose)

        datagen = create_triplet_generator_from_index_path(X,
                    index_path=self.annoy_index_path,
                    k=self.k,
                    batch_size=self.batch_size,
                    search_k=self.search_k,
                    precompute=self.precompute,
                    verbose=self.verbose)

        loss_monitor = 'loss'

        if self.model_ is None:       
            if type(self.model_def) is str:
                input_size = (X.shape[-1],)
                self.model_ = build_network(base_network(self.model_def, input_size), embedding_dims=self.embedding_dims)
            else:
                self.model_ = build_network(self.model_def, embedding_dims=self.embedding_dims)
            try:
                self.model_.compile(optimizer='adam', loss=triplet_loss(distance=self.distance, margin=self.margin))
            except KeyError:
                raise Exception('Loss function not implemented.')
        self.encoder = self.model_.layers[3]

        if self.verbose > 0:
            print('Training neural network')

        hist = self.model_.fit_generator(datagen, 
            steps_per_epoch=int(X.shape[0] / self.batch_size), 
            epochs=self.epochs, 
            callbacks=[EarlyStopping(monitor=loss_monitor, patience=self.n_epochs_without_progress)],            
            shuffle=shuffle_mode,
            workers=multiprocessing.cpu_count(),
            verbose=self.verbose)
        self.loss_history_ += hist.history['loss']

    def fit(self, X, shuffle_mode=True):
        """Fit an ivis model.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data to be embedded.
        
        Returns
        -------
        returns an instance of self
        """

        self._fit(X, shuffle_mode)
        return self

    def fit_transform(self, X, shuffle_mode=True):
        """Fit to data then transform

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data to be embedded.
        
        Returns
        -------
        X_new : transformed array, shape (n_samples, embedding_dims)
            Embedding of the new data in low-dimensional space.
        """

        self.fit(X, shuffle_mode)
        return self.transform(X)
        
    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.
                                                    
        Returns
        -------
        X_new : array, shape (n_samples, embedding_dims)
            Embedding of the new data in low-dimensional space.
        """

        embedding = self.encoder.predict(X, verbose=self.verbose)
        return embedding

    def save_model(self, folder_path):
        """Save an ivis model

        Parameters
        ----------
        folder_path : string
            Path to serialised model files and metadata
        """
        os.makedirs(folder_path) 
        # serialize weights to HDF5
        self.model_.save(os.path.join(folder_path, 'ivis_model.h5'))

        json.dump(self.__getstate__(), open(os.path.join(folder_path, 'ivis_params.json'), 'w'))
    
    def load_model(self, folder_path):
        """Load ivis model

        Parameters
        ----------
        folder_path : string
            Path to serialised model files and metadata

        Returns
        -------
        returns an ivis instance
        """ 

        ivis_config = json.load(open(os.path.join(folder_path,
            'ivis_params.json'), 'r'))
        self.__dict__ = ivis_config
        
        loss_function = triplet_loss(self.distance, self.margin)
        self.model_ = load_model(os.path.join(folder_path, 'ivis_model.h5'),
                custom_objects={'tf':tf, loss_function.__name__ : loss_function })
        self.encoder = self.model_.layers[3]
        self.encoder._make_predict_function()
        return self
    
