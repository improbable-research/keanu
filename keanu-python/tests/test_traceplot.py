from keanu.plots import traceplot, make_1d
from keanu import BayesNet, Model
from keanu.vertex import Gamma, Gaussian
from keanu.algorithm import sample
from keanu.vartypes import sample_types
from numpy import array, array_equal
import pytest
from typing import Any


def test_make_1d():
    arr = array([[1, 2], [3, 4]])
    arr_1d = make_1d(arr)

    assert (arr_1d == [1, 2, 3, 4]).all()


@pytest.fixture
def trace() -> sample_types:
    gamma = Gamma(array([[1., 2.], [3., 4.]]), array([[1., 2.], [3., 4.]]))
    gaussian = Gaussian(gamma, 1.)

    gamma.set_label("gamma")

    trace = {
        gamma: [array([[1., 2.], [3., 4.]]), array([[2., 3.], [4., 5.]])],
        gaussian: [array([[0.1, 0.2], [0.3, 0.4]]), array([[0.2, 0.3], [0.4, 0.5]])]
    }

    return trace


# suppress matplotlib warning of use of non-gui backend
@pytest.mark.filterwarnings("ignore:Matplotlib")
def test_traceplot_returns_axeplot_with_correct_data(trace: sample_types) -> None:
    ax = traceplot(trace)

    gamma_ax = ax[0][0]
    gaussian_ax = ax[1][0]

    assert gamma_ax.get_title() == 'gamma'
    assert gaussian_ax.get_title() == "[GaussianVertex => <class 'keanu.vertex.base.Double'>]"

    gamma_lines = gamma_ax.get_lines()
    gaussian_lines = gaussian_ax.get_lines()

    gamma_ax_data = [l.get_ydata() for l in gamma_lines]
    gaussian_ax_data = [l.get_ydata() for l in gaussian_lines]

    assert array_equal(gamma_ax_data, [array([1., 2.]), array([2., 3.]), array([3., 4.]), array([4., 5.])])
    assert array_equal(gaussian_ax_data, [array([0.1, 0.2]), array([0.2, 0.3]), array([0.3, 0.4]), array([0.4, 0.5])])


@pytest.mark.mpl_image_compare(filename='test_traceplot_generates_correct_image.png')
def test_traceplot_generates_correct_image(trace: sample_types) -> Any:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, squeeze=False)
    ax = traceplot(trace, ax=ax)

    return fig
