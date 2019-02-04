import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from keanu import stats
from typing import Tuple, Any, List
from keanu.vartypes import primitive_types
from math import log, floor
from numpy import ndarray


def __create_new_mpl() -> Tuple[Any, Any]:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return fig, ax


def __plot_corr(ax: Any, acf_x: ndarray, nlags: int, **kwargs: Any) -> None:
    ax.vlines(np.arange(nlags), [0], acf_x)
    ax.axhline(0, 0, 1, linewidth=1, color='black')
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 5)
    kwargs.setdefault('linestyle', 'None')
    ax.set_title("Autocorrelation")
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(acf_x[:nlags], **kwargs)


def __calc_max_lag(data_len: int) -> int:
    lim = min(int(floor(10 * log(data_len, 10))), data_len - 1)
    return lim


def plot_acf(data: List[primitive_types], nlags: int = None) -> Any:
    autocorr = stats.autocorrelation(data)
    fig, ax = __create_new_mpl()
    if nlags is None:
        nlags = __calc_max_lag(len(autocorr))
    __plot_corr(ax, autocorr, nlags)
    return fig
