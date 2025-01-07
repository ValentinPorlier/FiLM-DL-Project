"""This module contains useful functions for other modules.
"""

import numpy as np


def poly(x, order=3):
    """Evaluates the different powers of an input vector.

    The input vector is evaluated element-wise
    to the power 1, 2, ..., `order`. The resulting vectors
    are then concatenated and returned.

    Parameters
    ----------
    x: array_like
        The input vector, of shape `(n, 1)`.
    order: int
        The maximum order to which the powers of
        `x`are computed.

    Returns
    -------
    x_out: array_like
        The concatenation of all
        the powers of `x`, of shape `(n, order)`.

    """
    x_out = x
    for i in range(2, order + 1):
        x_out = np.concatenate((x_out, np.power(x, i)), axis=1)
    return x_out
