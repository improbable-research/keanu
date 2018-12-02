from keanu.plots import traceplot, make_1d
from keanu import BayesNet
from keanu.vertex import Gamma, Bernoulli
from keanu.algorithm import sample
from keanu.vartypes import runtime_numpy_types
from numpy import array, array_equal
import pytest


def test_make_1d():
    arr = array([[1, 2], [3, 4]])
    arr_1d = make_1d(arr)

    assert (arr_1d == [1, 2, 3, 4]).all()


# suppress matplotlib warning of use of PS as backend
@pytest.mark.filterwarnings("ignore:Matplotlib")
def test_traceplot_returns_axeplot_with_samples():
    gamma = Gamma(array([1., 2.]), array([1., 2.]))
    bernoulli = Bernoulli(gamma)

    gamma.set_label("gamma")
    bernoulli.set_label("bernoulli")

    net = BayesNet(bernoulli.get_connected_graph())

    trace = sample(net=net, sample_from=net.get_latent_vertices(), draws=2)
    ax = traceplot(trace)

    assert ax[0][0].get_title() == 'gamma'
    assert ax[1][0].get_title() == 'bernoulli'
    assert_ax_equals_trace_with_two_vertices(ax, trace[gamma], trace[bernoulli])


def assert_ax_equals_trace_with_two_vertices(ax, vertex1_trace, vertex2_trace):
    assert len(ax) == 2

    # each line in a plot corresponds to an element in sample ndarray
    lines1 = ax[0][0].get_lines()
    lines2 = ax[1][0].get_lines()

    yd1 = [array([lines1[0].get_ydata()[0], lines1[1].get_ydata()[0]]), array([lines1[0].get_ydata()[1], lines1[1].get_ydata()[1]])]
    yd2 = [array([lines2[0].get_ydata()[0], lines2[1].get_ydata()[0]]), array([lines2[0].get_ydata()[1], lines2[1].get_ydata()[1]])]

    assert (array_equal(yd1, vertex1_trace) and array_equal(yd2, vertex2_trace)) or (array_equal(yd1, vertex2_trace) and array_equal(yd2, vertex1_trace))
