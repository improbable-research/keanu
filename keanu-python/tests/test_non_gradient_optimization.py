import keanu as kn
from examples import thermometers
from keanu.context import KeanuContext
from py4j.java_gateway import get_field
import pytest


@pytest.fixture
def setup_model():
    a = kn.Gaussian(0., 50.)
    b = kn.Gaussian(0., 50.)
    c = a + b
    d = kn.Gaussian(c, 1.)
    d.observe(20.0)
    return (a, b)


def test_non_gradient_op_bayes_net():
    model = thermometers.model()
    net = kn.BayesNet(model.temperature.getConnectedGraph())
    gradient_optimizer = kn.NonGradientOptimizer(net)
    assert gradient_optimizer.net is net


def test_non_gradient_op_vertex():
    model = thermometers.model()
    gradient_optimizer = kn.NonGradientOptimizer(model.temperature)
    assert len(gradient_optimizer.net.vertices) == 7


def test_non_gradient_op_throws_with_invalid_net_param():
    with pytest.raises(ValueError) as excinfo:
        kn.NonGradientOptimizer(500)


def test_non_gradient_can_set_builder_properties():
    model = thermometers.model()
    non_gradient_optimizer = kn.NonGradientOptimizer(model.temperature, 5000, 10., 0.1, 0.2)


def test_map_non_gradient(setup_model):
    a, b = setup_model
    non_gradient_optimizer = kn.NonGradientOptimizer(a)
    logProb = non_gradient_optimizer.max_a_posteriori()
    assert logProb < 0.

    sum_ab = a.getValue().scalar() + b.getValue().scalar()
    assert 19.9 < sum_ab < 20.1


def test_max_likelihood_non_gradient(setup_model):
    a, b = setup_model
    non_gradient_optimizer = kn.NonGradientOptimizer(a)
    logProb = non_gradient_optimizer.max_likelihood()
    assert logProb < 0.

    sum_ab = a.getValue().scalar() + b.getValue().scalar()
    assert 19.9 < sum_ab < 20.1
