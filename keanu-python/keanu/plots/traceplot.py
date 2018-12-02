import matplotlib
matplotlib.use('TkAgg')
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

        x = range(x0, x0 + len(data))
        ax[index][0].set_xticks(x)
        ax[index][0].plot(x, data)

        plt.pause(0.1)

    return ax


def make_1d(a: numpy_types) -> numpy_types:
    a = np.atleast_1d(a)
    # collapse array in row-major (C-style) order
    # see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html
    return a.flatten(order='C')
