""" scikit-learn wrapper class for the Ivis algorithm. """

from data.triplet_generators import create_triplet_generator
from nn.network import build_network, selu_base_network
from nn.losses import triplet_loss

from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.base import BaseEstimator


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
        The number of neighbours to retrieve for each point

    distance : string, optional (default: "pn")
        The loss function used to train the neural network. One of "pn", "euclidean", "softmax_ratio_pn", "softmax_ratio".
    
    batch_size : int, optional (default: 128)
        The size of mini-batches used during gradient descent while training the neural network.

    epochs : int, optional (default: 10000)
        The maximum number of epochs to train the model for. Each epoch the network will see a triplet based on each data-point once.

    n_epochs_without_progress: int, optional (default: 50)
        After n number of epochs without an improvement to the loss, terminate training early.

    margin: float, optional (default: 1)
        The distance that is enforced between points by the triplet loss functions

    ntrees: int, optional (default: 50)
        The number of random projections trees built by Annoy to approximate KNN. The more trees the higher the memory usage, but the better the accuracy of results.

    search_k: int, optional (default: -1)
        The maximum number of nodes inspected during a nearest neighbour query by Annoy. The higher, the more computation time required, but the higher the accuracy. The default 
        is n_trees * k, where k is the number of neighbours to retrieve. If this is set too low, a variable number of neighbours may be retrieved per data-point.

    precompute : boolean, optional (default: True)
        Whether to pre-compute the nearest neighbours. Pre-computing is significantly faster, but requires more memory. If memory is limited, try setting this to False.
    

    Attributes
    ----------
    model_ : keras Model 
        Stores the trained neural network model mapping inputs to embeddings
    
    loss_history_ : array-like, floats
        The loss history at the end of each epoch during training of the model.

    
    """

    def __init__(self, embedding_dims=2, k=150, distance='pn', batch_size=128, epochs=1000, n_epochs_without_progress=50, margin=1, ntrees=50, search_k=-1, precompute=True):
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

    def _fit(self, X):
        input_size = (X.shape[-1],)
        datagen = create_triplet_generator(X, k=self.k, ntrees=self.ntrees, batch_size=self.batch_size, search_k=self.search_k, precompute=self.precompute)

        try:
            model = build_network(selu_base_network(input_size))
            model.compile(optimizer='adam', loss=triplet_loss(distance=self.distance, margin=self.margin))
        except KeyError:
            raise Exception('Loss function not implemented.')
        
        hist = model.fit_generator(datagen, steps_per_epoch=int(X.shape[0] / self.batch_size), epochs=self.epochs, callbacks=[EarlyStopping(monitor='loss', patience=50)] )
        self.loss_history_ = hist.history['loss']
        self.model_ = model.layers[3]

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
        
    def transform(self, X):
        embedding = self.model_.predict(X)
        return embedding

    def save(self, filepath):
        self.model_.save(filepath)
    
    def load(self, filepath):
        model = load_model(filepath)
        self.model_ = model
        return self