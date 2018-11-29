import matplotlib
matplotlib.use('PS') # backend without gui, see: https://stackoverflow.com/a/50200567
import matplotlib.pyplot as plt
import numpy as np


def traceplot(trace, ax=None, x0=0):
    # TODO: allow user to specify vertex ids or labels or vertex
    vertex_ids = trace.keys()

    if ax is None:
        _, ax = plt.subplots(len(vertex_ids), 1, squeeze=False)

    for index, vertex_id in enumerate(vertex_ids):
        data = [make_1d(v) for v in trace[vertex_id]]
        size = len(data)

        # TODO: use label for title ax[index][0].set_title(vertex_id)
        ax[index][0].plot(range(x0, x0 + size), data)
        plt.pause(0.1)

    return ax


def make_1d(a):
    a = np.atleast_1d(a)
    return a.flatten()
