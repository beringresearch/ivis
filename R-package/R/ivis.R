#' IVIS algorithm
#'
#' @param embedding_dims: Number of dimensions in the embedding space
#' @param k:          The number of neighbours to retrieve for each point.
#'                    Must be less than one minus the number of rows in the
#'                    dataset.
#' @param distance:   The loss function used to train the neural network.
#'                    One of "pn", "euclidean", "manhattan_pn", "manhattan",
#'                    "chebyshev", "chebyshev_pn", "softmax_ratio_pn",
#'                    "softmax_ratio".
#' @param batch_size: The size of mini-batches used during gradient descent
#'                    while training the neural network. Must be less than
#'                    the num_rows in the dataset.
#' @param epochs:     The maximum number of epochs to train the model for.
#'                    Each epoch the network will see a triplet based on each
#'                    data-point once.
#' @param n_epochs_without_progress: After n number of epochs without an
#'                    improvement to the loss, terminate training early.
#' @param margin:     The distance that is enforced between points by the
#'                    triplet loss functions.
#' @param ntrees:     The number of random projections trees built by Annoy to
#'                    approximate KNN. The more trees the higher the memory
#'                    usage, but the better the accuracy of results.
#' @param search_k:   The maximum number of nodes inspected during a nearest
#'                    neighbour query by Annoy. The higher, the more
#'                    computation time required, but the higher the accuracy.
#'                    The default is n_trees * k, where k is the number of
#'                    neighbours to retrieve. If this is set too low, a
#'                    variable number of neighbours may be retrieved per
#'                    data-point.
#' @param precompute: Whether to pre-compute the nearest neighbours.
#'                    Pre-computing is significantly faster, but requires
#'                    more memory. If memory is limited, try setting this to
#'                    False.
#' @param model:      str or keras.models.Model. The keras model to train
#'                    using triplet loss. If a model object is provided, an
#'                    embedding layer of size 'embedding_dims' will be
#'                    appended to the end of the network. If a string, a
#'                    pre-defined network by that name will be used.
#'                    Possible options are: 'default', 'hinton', 'maaten'.
#'                    By default, a selu network composed of 3 dense layers
#'                    of 128 neurons each will be created, followed by an
#'                    embedding layer of size 'embedding_dims'.
#' @param supervision_metric: str or function. The supervision metric to
#'                    optimize when training keras in supervised mode. Supports all of the
#'                    classification or regression losses included with keras, so long as
#'                    the labels are provided in the correct format. A list of keras' loss
#'                    functions can be found at https://keras.io/losses/ .
#' @param supervision_weight: Float between 0 and 1 denoting the
#'                    weighting to give to classification vs triplet loss
#'                    when training in supervised mode. The higher the
#'                    weight, the more classification influences training.
#                     Ignored if using Ivis in unsupervised mode.
#' @param annoy_index_path: The filepath of a pre-trained annoy index file
#'                    saved on disk. If provided, the annoy index file will
#'                    be used. Otherwise, a new index will be generated and
#'                    saved to disk in the current directory as
#'                    'annoy.index'.
#' @param eager_execution: Whether to use eager execution with TensorFlow.
#                     Disabled by default, as training is much faster with this option off. 
#' @param verbose:    Controls the volume of logging output the model
#'                    produces when training. When set to 0, silences
#'                    outputs, when above 0 will print outputs.
#' @export

ivis <- function(embedding_dims = 2L,
    k = 150L,
    distance = "pn",
    batch_size = 128L,
    epochs = 1000L,
    n_epochs_without_progress = 50L,
    margin = 1,
    ntrees = 50L,
    search_k = -1L,
    precompute = TRUE,
    model = "default",
    supervision_metric = "sparse_categorical_crossentropy",
    supervision_weight = 0.5,
    eager_execution = FALSE,
    annoy_index_path=NULL, verbose=1L){


    #X <- data.matrix(X)
    k <- as.integer(k)
    
    embedding_dims <- as.integer(embedding_dims)
    batch_size <- as.integer(batch_size)
    epochs <- as.integer(epochs)
    n_epochs_without_progress = as.integer(n_epochs_without_progress)
    ntrees <- as.integer(ntrees)
    search_k <- as.integer(search_k)

    model <- ivis_object$Ivis(embedding_dims=embedding_dims,
                              k=k, distance=distance, batch_size=batch_size,
                              epochs=epochs,
                              n_epochs_without_progress=n_epochs_without_progress,
                              margin=margin, ntrees=ntrees,
                              search_k=search_k,
                              precompute=precompute, model=model,
                              supervision_metric=supervision_metric,
                              supervision_weight=supervision_weight,
                              annoy_index_path=annoy_index_path,
                              eager_execution=eager_execution,
                              verbose=verbose)
  
    return(model)

    }
