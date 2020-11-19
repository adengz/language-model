from typing import Tuple


def count_params(model) -> int:
    """
    Counts trainable parameters in a model.

    Args:
        model: Model.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_time(start: float, end: float) -> Tuple[float, float]:
    """
    Calculates elapsed times in minutes and seconds.

    Args:
        start: Starting time.
        end: Ending time.

    Returns:
        minutes, seconds.
    """
    elapsed = end - start
    mins = int(elapsed / 60)
    secs = int(elapsed - mins * 60)
    return mins, secs
