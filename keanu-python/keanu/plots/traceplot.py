import matplotlib
matplotlib.use('PS') # backend without gui, see: https://stackoverflow.com/a/50200567
import matplotlib.pyplot as plt


def traceplot(trace, ax=None, x0=0):
    vertex_ids = trace.keys()

    if ax is None:
        _, ax = plt.subplots(len(vertex_ids), 1, squeeze=False)

    for index, vertex_id in enumerate(vertex_ids):
        data = trace[vertex_id]
        size = len(data) if hasattr(data, '__len__') else 1
        ax[index][0].plot(range(x0, x0 + size), data)

    return ax
