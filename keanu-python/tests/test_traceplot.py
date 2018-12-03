from keanu.plots import traceplot
from keanu.vartypes import sample_types
from keanu import Model
from numpy import array
from numpy.testing import assert_array_equal
import pytest
from typing import Any
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


@pytest.fixture
def trace() -> sample_types:
    with Model() as m:
        trace = {
            "gamma": [array([[1., 2.], [3., 4.]]), array([[2., 3.], [4., 5.]])],
            "gaussian": [array([[0.1, 0.2], [0.3, 0.4]]), array([[0.2, 0.3], [0.4, 0.5]])]
        }

    return trace


def test_traceplot_returns_axeplot_with_correct_data(trace: sample_types) -> None:
    ax = traceplot(trace, labels=['gamma', 'gaussian'])

    gamma_ax = ax[0][0]
    gaussian_ax = ax[1][0]

    assert gamma_ax.get_title() == 'gamma'
    assert gaussian_ax.get_title() == 'gaussian'

    gamma_lines = gamma_ax.get_lines()
    gaussian_lines = gaussian_ax.get_lines()

    gamma_ax_data = [l.get_ydata() for l in gamma_lines]
    gaussian_ax_data = [l.get_ydata() for l in gaussian_lines]

    assert_array_equal(gamma_ax_data, [array([1., 2.]), array([2., 3.]), array([3., 4.]), array([4., 5.])])
    assert_array_equal(gaussian_ax_data, [array([0.1, 0.2]), array([0.2, 0.3]), array([0.3, 0.4]), array([0.4, 0.5])])


@pytest.mark.mpl_image_compare(filename='test_traceplot_generates_correct_image.png')
def test_traceplot_generates_correct_image(trace: sample_types) -> Any:
    fig, ax = plt.subplots(2, 1, squeeze=False)
    ax = traceplot(trace, ax=ax)

    return fig
