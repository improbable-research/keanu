import matplotlib
matplotlib.use('PS') # see matplotlib backends: https://stackoverflow.com/a/50200567
import matplotlib.pyplot as plt

import numpy as np
from keanu.vartypes import sample_types, numpy_types
from typing import Any


def traceplot(trace: sample_types,
              ax: Any = None,
              x0: int = 0) -> Any:
    vertices = trace.keys()

    if ax is None:
        _, ax = plt.subplots(len(vertices), 1, squeeze=False)

    for index, vertex in enumerate(vertices):
        data = [make_1d(v) for v in trace[vertex]]

        label = vertex.get_label()
        # Vertex#__repr__ is explicitly passed to avoid usage of overloaded operations in third party libraries
        ax[index][0].set_title(vertex.__repr__() if label is None else label)

        ax[index][0].plot(range(x0, x0 + len(data)), data)
        plt.pause(0.1)

    return ax


def make_1d(a: numpy_types) -> numpy_types:
    a = np.atleast_1d(a)
    # collapse array in row-major (C-style) order
    # see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html
    return a.flatten(order='C')
