try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:  # mpl is optional
    pass

import numpy as np
from keanu.vartypes import sample_types, numpy_types
from typing import Any, List, Tuple, Union


def traceplot(trace: sample_types, labels: List[Union[str, Tuple[str, str]]] = None, ax: Any = None,
              x0: int = 0) -> Any:
    """
    Plot samples values.

    :param trace:  result of MCMC run
    :param labels: labels of vertices to be plotted. if None, all vertices are plotted.
    :param ax: Matplotlib axes
    :param x0: index of first data point, used for sample stream plots
    """

    if labels is None:
        labels = list(trace.keys())

    if ax is None:
        _, ax = plt.subplots(len(labels), 1, squeeze=False)

    for index, label in enumerate(labels):
        data = [sample for sample in trace[label]]

        ax[index][0].set_title(label)
        ax[index][0].plot(__integer_xaxis(ax[index][0], x0, len(data)), data)

    __pause_for_crude_animation()

    return ax


def __integer_xaxis(ax: Any, x0: int, n: int) -> range:
    x = range(x0, x0 + n)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return x


def __pause_for_crude_animation() -> None:
    plt.pause(0.1)
