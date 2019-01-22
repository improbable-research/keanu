import matplotlib.pyplot as plt

from keanu.plots import traceplot
from keanu.vartypes import sample_types
from keanu.vertex import VertexLabel
from numpy import array
from numpy.testing import assert_array_equal
import pytest
from typing import Any
from collections import OrderedDict


@pytest.fixture
def gamma_label() -> VertexLabel:
    return VertexLabel("gamma")


@pytest.fixture
def gaussian_label() -> VertexLabel:
    return VertexLabel("gaussian")


@pytest.fixture
def trace(gamma_label: VertexLabel, gaussian_label: VertexLabel) -> sample_types:
    return OrderedDict([(gamma_label, [array([[1., 2.], [3., 4.]]),
                                       array([[2., 3.], [4., 5.]])]),
                        (gaussian_label, [array([[0.1, 0.2], [0.3, 0.4]]),
                                          array([[0.2, 0.3], [0.4, 0.5]])])])


def test_traceplot_returns_axesplot_with_correct_data(gamma_label: VertexLabel, gaussian_label: VertexLabel,
                                                      trace: sample_types) -> None:
    ax = traceplot(trace, labels=[gamma_label, gaussian_label])

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
