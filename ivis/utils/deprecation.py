import warnings


def check_deprecated_ntrees(ntrees):
    """ Checks for the use of ntrees """
    if ntrees is not None:
        warnings.warn("Received a value for `ntrees`. In the future, passing "
                      "a value to `ntrees` will result in a ValueError. "
                      "`n_trees` was set to `ntrees`. Use `n_trees` "
                      "instead.", FutureWarning)
