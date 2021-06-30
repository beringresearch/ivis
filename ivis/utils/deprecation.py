from functools import wraps
from inspect import Parameter, signature
from warnings import warn


def check_deprecated_ntrees(ntrees, *, version="2.2"):
    """ Checks for the use of ntrees """
    if ntrees is not None:
        warn(f"Received a value for `ntrees`. From version {version}, "
             "passing a value to `ntrees` will result in a ValueError. "
             f"`n_trees` was set to `{ntrees}`. Use `n_trees` instead.",
             FutureWarning)


def deprecate_positional_args(func=None, *, version="2.2"):
    """Decorator for methods that issues warnings for positional arguments.

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    This function was imported from sklearn.utils.validation to have the
    same behavior without using the function from sklearn, which is private.

    Parameters
    ----------
    func : callable, default=None
        Function to check arguments on.
    version : callable, default="2.2"
        The version when positional arguments will result in error.
    """
    def inner_deprecate_positional_args(f):
        sig = signature(f)
        kwonly_args = []
        all_args = []

        for name, param in sig.parameters.items():
            if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
                all_args.append(name)
            elif param.kind == Parameter.KEYWORD_ONLY:
                kwonly_args.append(name)

        @wraps(f)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)

            # extra_args > 0
            args_msg = ['{}={}'.format(arg_name, arg_value)
                        for arg_name, arg_value in zip(kwonly_args[:extra_args],
                                                       args[-extra_args:])]
            args_msg = ", ".join(args_msg)
            warn(f"Pass {args_msg} as keyword args. From version {version}, "
                 f"passing these as positional arguments will result in an "
                 f"error", FutureWarning)
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)
        return inner_f

    if func is not None:
        return inner_deprecate_positional_args(func)

    return inner_deprecate_positional_args
