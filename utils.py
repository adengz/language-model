import time


def count_params(model):
    """
    Counts trainable parameters in a model.

    Args:
        model: Model.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def timeit(f):
    """
    Timer decorator for profiling.

    Args:
        f: Decorated function.

    Returns:
        Whatever returned by invoking f.
    """
    def timed(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        mins = int(elapsed / 60)
        secs = elapsed - mins * 60
        print(f'\tElapsed time: {mins}m {secs: .3f}s')

        return ret
    return timed
