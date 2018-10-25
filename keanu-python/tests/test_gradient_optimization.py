import pytest
import keanu as kn
from examples import thermometers
from keanu.context import KeanuContext
from py4j.java_gateway import get_field
from py4j.protocol import Py4JJavaError


@pytest.fixture
def model():
    model = thermometers.model()
    model.thermometer_one.observe(22.0)
    model.thermometer_two.observe(20.0)
    return model


def test_gradient_op_bayes_net(model):
    net = kn.BayesNet(model.temperature.getConnectedGraph())
    gradient_optimizer = kn.GradientOptimizer(net)
    assert gradient_optimizer.net is net


def test_gradient_op_vertex(model):
    gradient_optimizer = kn.GradientOptimizer(model.temperature)
    assert len(gradient_optimizer.net.getLatentVertices()) == 1


def test_gradient_op_throws_with_invalid_net_param():
    with pytest.raises(TypeError) as excinfo:
        kn.GradientOptimizer(500)


def test_gradient_can_set_max_eval_builder_properties(model):
    net = kn.BayesNet(model.temperature.getConnectedGraph())
    gradient_optimizer = kn.GradientOptimizer(model.temperature, max_evaluations=5)

    with pytest.raises(Py4JJavaError) as excinfo:
        #This throws a Gradient Optimizer: "Reached Max Evaluations" error
        logProb = gradient_optimizer.max_a_posteriori()


def test_thermometers_map_gradient(model):
    net = kn.BayesNet(model.temperature.getConnectedGraph())
    gradient_optimizer = kn.GradientOptimizer(net)
    logProb = gradient_optimizer.max_a_posteriori()
    assert logProb < 0.

    temperature = model.temperature.getValue().scalar()
    assert 20.995 < temperature <  21.005


def test_thermometers_max_likelihood_gradient(model):
    net = kn.BayesNet(model.temperature.getConnectedGraph())
    gradient_optimizer = kn.GradientOptimizer(net)
    logProb = gradient_optimizer.max_likelihood()
    assert logProb < 0.

    temperature = model.temperature.getValue().scalar()
    assert 20.995 < temperature <  21.005
