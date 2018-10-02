#' IVIS algorithm
#'
#' @param X numerical matrix to be reduced. Columns correspond to features.
#' @param y int, optional (default: NULL). Optional class vector triggering supervised tripplet selection.
#' @param embedding_dims int, optional (default: 2) Number of dimensions in the embedding space
#' @param k int, optional (default: 150)
#'        The number of neighbours to retrieve for each point
#' @param distance string, optional (default: "pn")
#'        The loss function used to train the neural network. One of "pn", "euclidean", "softmax_ratio_pn", "softmax_ratio". 
#' @param batch_size int, optional (default: 128)
#'        The size of mini-batches used during gradient descent while training the neural network.
#' @param epochs int, optional (default: 1000)
#'        The maximum number of epochs to train the model for. Each epoch the network will see a triplet based on each data-point once.
#' @param n_epochs_without_progress int, optional (default: 50)
#'        After n number of epochs without an improvement to the loss, terminate training early.
#' @param margin float, optional (default: 1)
#'        The distance that is enforced between points by the triplet loss functions
#' @param ntrees int, optional (default: 50)
#'        The number of random projections trees built by Annoy to approximate KNN. The more trees the higher the memory usage, but the better the accuracy of results.
#' @param search_k int, optional (default: -1)
#'        The maximum number of nodes inspected during a nearest neighbour query by Annoy. The higher, the more computation time required, but the higher the accuracy. The default 
#'        is n_trees * k, where k is the number of neighbours to retrieve. If this is set too low, a variable number of neighbours may be retrieved per data-point.
#' @param precompute boolean, optional (default: True)
#'        Whether to pre-compute the nearest neighbours. Pre-computing is significantly faster, but requires more memory. If memory is limited, try setting this to False.
#' @export

ivis <- function(X, y = NULL, embedding_dims = 2L,
    k = 150L,
    distance = "pn",
    batch_size = 128L,
    epochs = 1000L,
    n_epochs_without_progress = 50L,
    margin = 1,
    ntrees = 50L,
    search_k = -1L,
    precompute = TRUE){


    X <- data.matrix(X)
    if (!is.null(y)) y <- as.integer(y)
    embedding_dims <- as.integer(embedding_dims)
    batch_size <- as.integer(batch_size)
    epochs <- as.integer(epochs)
    n_epochs_without_progress = as.integer(n_epochs_without_progress)
    ntrees <- as.integer(ntrees)

    search_k <- as.integer(search_k)

    model <- ivis_object$Ivis(embedding_dims=embedding_dims,
        k = k, distance = distance, batch_size = batch_size,
        epochs = epochs, n_epochs_without_progress = n_epochs_without_progress,
        margin = margin, ntrees = ntrees, search_k = search_k, precompute = precompute)
    
    embeddings = model$fit_transform(X = X, y = y)
    return(embeddings)

    }
