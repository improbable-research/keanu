import pytest
import keanu as kn
from examples import thermometers
from keanu.context import KeanuContext
from py4j.java_gateway import get_field
from py4j.protocol import Py4JJavaError

@pytest.fixture
def model():
    kn.KeanuRandom().set_default_random_seed(1)
    with kn.Model() as m:
        m.a = kn.Gaussian(0., 50.)
        m.b = kn.Gaussian(0., 50.)
        m.c = m.a + m.b
        m.d = kn.Gaussian(m.c, 1.)
        m.d.observe(20.0)
    return m


def test_non_gradient_op_bayes_net(model):
    net = kn.BayesNet(model.a.get_connected_graph())
    gradient_optimizer = kn.NonGradientOptimizer(net)
    assert gradient_optimizer.net is net


def test_non_gradient_op_vertex(model):
    non_gradient_optimizer = kn.NonGradientOptimizer(model.a)
    assert len(list(non_gradient_optimizer.net.get_latent_vertices())) == 2


def test_non_gradient_op_throws_with_invalid_net_param():
    with pytest.raises(TypeError) as excinfo:
        kn.NonGradientOptimizer(500)


def test_non_gradient_can_set_max_eval_builder_properties(model):
    non_gradient_optimizer = kn.NonGradientOptimizer(model.a, max_evaluations=5)

    with pytest.raises(Py4JJavaError) as excinfo:
        #This throws a Gradient Optimizer: "Reached Max Evaluations" error
        logProb = non_gradient_optimizer.max_a_posteriori()


def test_non_gradient_can_set_bounds_range_builder_properties(model):
    non_gradient_optimizer = kn.NonGradientOptimizer(model.a, bounds_range=0.1)
    logProb = non_gradient_optimizer.max_a_posteriori()

    sum_ab = model.a.get_value() + model.b.get_value()
    assert not (19.9 < sum_ab < 20.1)


def test_map_non_gradient(model):
    non_gradient_optimizer = kn.NonGradientOptimizer(model.a)
    logProb = non_gradient_optimizer.max_a_posteriori()
    assert logProb < 0.

    sum_ab = model.a.get_value() + model.b.get_value()
    assert 19.9 < sum_ab < 20.1


def test_max_likelihood_non_gradient(model):
    non_gradient_optimizer = kn.NonGradientOptimizer(model.a)
    logProb = non_gradient_optimizer.max_likelihood()
    assert logProb < 0.

    sum_ab = model.a.get_value() + model.b.get_value()
    assert 19.9 < sum_ab < 20.1
