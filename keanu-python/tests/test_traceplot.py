import matplotlib.pyplot as plt

from keanu.plots import traceplot
from keanu.vartypes import sample_types
from numpy import array
from numpy.testing import assert_array_equal
import pytest
from typing import Any
from collections import OrderedDict


@pytest.fixture
def trace() -> sample_types:
    return OrderedDict([("gamma", [1., 2., 3., 4.]), ("gaussian", [0.1, 0.2, 0.3, 0.4])])


def test_traceplot_returns_axesplot_with_correct_data(trace: sample_types) -> None:
    ax = traceplot(trace, labels=['gamma', 'gaussian'])

    gamma_ax = ax[0][0]
    gaussian_ax = ax[1][0]

    assert gamma_ax.get_title() == 'gamma'
    assert gaussian_ax.get_title() == 'gaussian'

    gamma_lines = gamma_ax.get_lines()
    gaussian_lines = gaussian_ax.get_lines()

    gamma_ax_data = [l.get_ydata() for l in gamma_lines]
    gaussian_ax_data = [l.get_ydata() for l in gaussian_lines]

    assert_array_equal(gamma_ax_data, [[1., 2., 3., 4.]])
    assert_array_equal(gaussian_ax_data, [[0.1, 0.2, 0.3, 0.4]])
