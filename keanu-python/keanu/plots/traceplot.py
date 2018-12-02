import matplotlib
matplotlib.use('TkAgg') # see: https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
import matplotlib.pyplot as plt

import numpy as np
from keanu.vartypes import sample_types, numpy_types
from typing import Any, List


def traceplot(trace: sample_types, labels: List[str] = None, ax: Any = None, x0: int = 0) -> Any:
    if labels is None:
        labels = list(trace.keys())

    if ax is None:
        _, ax = plt.subplots(len(labels), 1, squeeze=False)

    for index, label in enumerate(labels):
        data = [make_1d(v) for v in trace[label]]

        ax[index][0].set_title(label)
        ax[index][0].plot(__discrete_xaxis(ax[index][0], x0, len(data)), data)

        __pause_for_crude_animation()

    return ax


def make_1d(a: numpy_types) -> numpy_types:
    a = np.atleast_1d(a)
    # collapse array in row-major (C-style) order
    # see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html
    return a.flatten(order='C')


def __discrete_xaxis(ax: Any, x0: int, n: int) -> range:
    x = range(x0, x0 + n)
    ax.set_xticks(x)
    return x


def __pause_for_crude_animation():
    plt.pause(0.1)
