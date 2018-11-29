#try:
import matplotlib
matplotlib.use('PS') # without gui
import matplotlib.pyplot as plt
#except ImportError:
#    matplotlib = None
    # matplotlib is optional. see: https://github.com/scikit-optimize/scikit-optimize/issues/637#issuecomment-371909456


def traceplot(trace):
    vertex_ids = trace.keys()

    _, ax = plt.subplots(len(vertex_ids), 1, squeeze=False)

    for index, vertex_id in enumerate(vertex_ids):
        ax[index][0].plot(trace[vertex_id])

    return ax
