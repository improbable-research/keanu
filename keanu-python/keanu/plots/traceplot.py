import matplotlib
matplotlib.use('PS') # see matplotlib backends: https://stackoverflow.com/a/50200567
import matplotlib.pyplot as plt
import numpy as np


def traceplot(trace, ax=None, x0=0):
    vertices = trace.keys()

    if ax is None:
        _, ax = plt.subplots(len(vertices), 1, squeeze=False)

    for index, vertex in enumerate(vertices):
        data = [make_1d(v) for v in trace[vertex]]

        # TODO: set label as title
        ax[index][0].plot(range(x0, x0 + len(data)), data)
        plt.pause(0.1)

    return ax


def make_1d(a):
    a = np.atleast_1d(a)
    # collapse array in row-major (C-style) order
    # see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html
    return a.flatten(order='C')
