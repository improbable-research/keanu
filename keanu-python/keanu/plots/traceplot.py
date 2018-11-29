import matplotlib
matplotlib.use('PS') # backend without gui, see: https://stackoverflow.com/a/50200567
import matplotlib.pyplot as plt


def traceplot(trace, ax=None):
    if ax is None:
        _, ax = plt.subplots(len(vertex_ids), 1, squeeze=False)

    for index, vertex_id in enumerate(vertex_ids):
        ax[index][0].plot(trace[vertex_id])

    return ax
