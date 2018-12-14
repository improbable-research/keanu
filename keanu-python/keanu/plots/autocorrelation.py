import numpy as np
from keanu import stats
from typing import Tuple, Any


def _import_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except:
        raise ImportError("Could not find Matplotlib")
    return plt


def _create_new_mpl() -> Tuple[Any, Any]:
    plt = _import_matplotlib()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return fig, ax


def _plot_corr(ax, acf_x, nlags, **kwargs) -> None:
    ax.vlines(np.arange(nlags), [0], acf_x)
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 5)
    kwargs.setdefault('linestyle', 'None')
    ax.set_title("Autocorrelation")
    ax.plot(acf_x[:nlags], **kwargs)


def _calc_max_lag(data_len) -> int:
    lim = min(int(np.floor(10 * np.log10(data_len))), data_len - 1)
    return lim


def plot_acf(data, nlags=None) -> None:
    autocorr = stats.autocorrelation(data)
    fig, ax = _create_new_mpl()
    if nlags is None:
        nlags = _calc_max_lag(len(autocorr))
    _plot_corr(ax, autocorr, nlags)
    return fig
