import matplotlib
matplotlib.use('PS') # backend without gui, see: https://stackoverflow.com/a/50200567
import matplotlib.pyplot as plt
import numpy as np


def traceplot(trace, ax=None, x0=0):
    # TODO: allow user to specify by labels or vertex
    vertices = trace.keys()

    if ax is None:
        # TODO: frequency plot or sample/stream plot?
        _, ax = plt.subplots(len(vertices), 1, squeeze=False)

    for index, vertex in enumerate(vertices):
        data = [make_1d(v) for v in trace[vertex]]
        size = len(data)

        # TODO: use label for title ax[index][0].set_title(vertex_id)
        ax[index][0].plot(range(x0, x0 + size), data)
        plt.pause(0.1)

    return ax


def make_1d(a):
    a = np.atleast_1d(a)
    return a.flatten()
