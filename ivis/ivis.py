""" scikit-learn wrapper class for the Ivis algorithm. """

from .data.triplet_generators import create_triplet_generator_from_index_path
from .nn.network import build_network, base_network
from .nn.losses import triplet_loss
from .data.knn import build_annoy_index

from keras.callbacks import EarlyStopping
from keras.models import load_model, Model
from sklearn.base import BaseEstimator
from annoy import AnnoyIndex
import json
import os
import multiprocessing


class Ivis(BaseEstimator):
    """
    Ivis is a technique that uses an artificial neural network for dimensionality reduction, often useful for the purposes of visualization.  
    The network trains on triplets of data-points at a time and pulls positive points together, while pushing more distant points away from each other.  
    Triplets are sampled from the original data using KNN aproximation using the Annoy library.

    Parameters
    ----------
    embedding_dims : int, optional (default: 2)
        Number of dimensions in the embedding space
    
    k : int, optional (default: 150)
        The number of neighbours to retrieve for each point. Must be less than one minus the number of rows in the dataset.

    distance : string, optional (default: "pn")
        The loss function used to train the neural network. One of "pn", "euclidean", "manhattan_pn", "manhattan", "chebyshev", "chebyshev_pn", "softmax_ratio_pn", "softmax_ratio".
    
    batch_size : int, optional (default: 128)
        The size of mini-batches used during gradient descent while training the neural network. Must be less than the num_rows in the dataset.

    epochs : int, optional (default: 1000)
        The maximum number of epochs to train the model for. Each epoch the network will see a triplet based on each data-point once.

    n_epochs_without_progress : int, optional (default: 50)
        After n number of epochs without an improvement to the loss, terminate training early.

    margin : float, optional (default: 1)
        The distance that is enforced between points by the triplet loss functions.

    ntrees : int, optional (default: 50)
        The number of random projections trees built by Annoy to approximate KNN. The more trees the higher the memory usage, but the better the accuracy of results.

    search_k : int, optional (default: -1)
        The maximum number of nodes inspected during a nearest neighbour query by Annoy. The higher, the more computation time required, but the higher the accuracy. The default 
        is n_trees * k, where k is the number of neighbours to retrieve. If this is set too low, a variable number of neighbours may be retrieved per data-point.

    precompute : boolean, optional (default: True)
        Whether to pre-compute the nearest neighbours. Pre-computing is significantly faster, but requires more memory. If memory is limited, try setting this to False.
    
    model: str or keras.models.Model (default: 'default')
        The keras model to train using triplet loss. If a model object is provided, an embedding layer of size 'embedding_dims' will be appended to the end of the network. 
        If a string, a pre-defined network by that name will be used. Possible options are: 'default', 'hinton', 'maaten'. 
        By default, a selu network composed of 3 dense layers of 128 neurons each will be created, followed by an embedding layer of size 'embedding_dims'.

    annoy_index_path: string, optional (default: None)
        The filepath of a pre-trained annoy index file saved on disk. If provided, the annoy index file will be used. Otherwise, a new index will be generated and saved to disk in the 
        current directory as 'annoy.index'.

    Attributes
    ----------
    model_ : keras Model 
        Stores the trained triplet loss neural network model. Can be used to resume training at a later stage.

    encoder : keras Model 
        Stores the encoder neural network model mapping inputs to embeddings. Used for inference.

    model_def: str or keras.models.Model
        The model definition provided to the Ivis object for the base network. Can be either a string or an actual model.

    loss_history_ : array-like, floats
        The loss history at the end of each epoch during training of the model.

    annoy_index_path : string
        The filepath of the annoy index currently in use by the model.
    
    """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

    def __init__(self, embedding_dims=2, k=150, distance='pn', batch_size=128, epochs=1000, n_epochs_without_progress=50, margin=1, ntrees=50, search_k=-1, precompute=True, model='default', annoy_index_path=None):
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

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['model_']
        del state['encoder']
        return state

    def _fit(self, X, shuffle_mode=True):
        
        if self.annoy_index_path is None:
            self.annoy_index_path = 'annoy.index'
            build_annoy_index(X, self.annoy_index_path, ntrees=self.ntrees)
        datagen = create_triplet_generator_from_index_path(X,
                    index_path=self.annoy_index_path,
                    k=self.k,
                    batch_size=self.batch_size,
                    search_k=self.search_k,
                    precompute=self.precompute)

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

        print('Training neural network')
        hist = self.model_.fit_generator(datagen, 
            steps_per_epoch=int(X.shape[0] / self.batch_size), 
            epochs=self.epochs, 
            callbacks=[EarlyStopping(monitor=loss_monitor, patience=self.n_epochs_without_progress)],            
            shuffle=shuffle_mode,
            workers=multiprocessing.cpu_count())
        self.loss_history_ += hist.history['loss']

    def fit(self, X, shuffle_mode=True):
        self._fit(X, shuffle_mode)
        return self

    def fit_transform(self, X, shuffle_mode=True):
        self.fit(X, shuffle_mode)
        return self.transform(X)
        
    def transform(self, X):
        embedding = self.encoder.predict(X)
        return embedding

    def save_model(self, folder_path):
        os.makedirs(folder_path)
        
        self.encoder.save(os.path.join(folder_path, 'ivis_model.h5'))
        json.dump(self.__getstate__(), open(os.path.join(folder_path, 'ivis.json'), 'w'))
    
    def load_model(self, folder_path):
        encoder = load_model(os.path.join(folder_path, 'ivis_model.h5'))
        ivis_config = json.load(open(os.path.join(folder_path,'ivis.json'), 'r'))
        self.__dict__ = ivis_config

        self.model_ = build_network(encoder, embedding_dims=self.embedding_dims)
        self.model_.compile(optimizer='adam', loss=triplet_loss(distance=self.distance, margin=self.margin))
        self.encoder = self.model_.layers[3]
        self.encoder._make_predict_function()
        return self
    