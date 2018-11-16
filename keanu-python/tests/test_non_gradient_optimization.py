import pytest
from py4j.protocol import Py4JJavaError
from keanu import KeanuRandom, Model, BayesNet
from keanu.vertex import Gaussian
from keanu.algorithm import NonGradientOptimizer


@pytest.fixture
def model():
    KeanuRandom.set_default_random_seed(1)
    with Model() as m:
        m.a = Gaussian(0., 50.)
        m.b = Gaussian(0., 50.)
        m.c = m.a + m.b
        m.d = Gaussian(m.c, 1.)
        m.d.observe(20.0)
    return m


def test_non_gradient_op_bayes_net(model):
    net = BayesNet(model.a.get_connected_graph())
    gradient_optimizer = NonGradientOptimizer(net)
    assert gradient_optimizer.net is net


def test_non_gradient_op_vertex(model):
    non_gradient_optimizer = NonGradientOptimizer(model.a)
    assert len(list(non_gradient_optimizer.net.get_latent_vertices())) == 2


def test_non_gradient_op_throws_with_invalid_net_param():
    with pytest.raises(TypeError):
        NonGradientOptimizer(500)


def test_non_gradient_can_set_max_eval_builder_properties(model):
    non_gradient_optimizer = NonGradientOptimizer(model.a, max_evaluations=5)

    with pytest.raises(Py4JJavaError):
        #This throws a Gradient Optimizer: "Reached Max Evaluations" error
        logProb = non_gradient_optimizer.max_a_posteriori()


def test_non_gradient_can_set_bounds_range_builder_properties(model):
    non_gradient_optimizer = NonGradientOptimizer(model.a, bounds_range=0.1)
    logProb = non_gradient_optimizer.max_a_posteriori()

    sum_ab = model.a.get_value() + model.b.get_value()
    assert not (19.9 < sum_ab < 20.1)


def test_map_non_gradient(model):
    non_gradient_optimizer = NonGradientOptimizer(model.a)
    logProb = non_gradient_optimizer.max_a_posteriori()
    assert logProb < 0.

    sum_ab = model.a.get_value() + model.b.get_value()
    assert 19.9 < sum_ab < 20.1


def test_max_likelihood_non_gradient(model):
    non_gradient_optimizer = NonGradientOptimizer(model.a)
    logProb = non_gradient_optimizer.max_likelihood()
    assert logProb < 0.

    sum_ab = model.a.get_value() + model.b.get_value()
    assert 19.9 < sum_ab < 20.1
