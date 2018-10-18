import keanu as kn
from examples import thermometers
from keanu.context import KeanuContext
from py4j.java_gateway import get_field
import pytest

@pytest.fixture
def thermometer_model():
    model = thermometers.model()
    model.thermometer_one.observe(22.0)
    model.thermometer_two.observe(20.0)
    return model

def test_gradient_op_bayes_net():
    model = thermometers.model()
    net = kn.BayesNet(model.temperature.getConnectedGraph())
    gradient_optimizer = kn.GradientOptimizer(net)
    assert gradient_optimizer.net is net


def test_gradient_op_vertex():
    model = thermometers.model()
    gradient_optimizer = kn.GradientOptimizer(model.temperature)
    assert len(gradient_optimizer.net.vertices) == 7


def test_gradient_op_throws_with_invalid_net_param():
    with pytest.raises(ValueError) as excinfo:
        kn.GradientOptimizer(500)


def test_gradient_can_set_builder_properties():
    model = thermometers.model()
    gradient_optimizer = kn.GradientOptimizer(model.temperature, 5000, 0.001, 0.002)


def test_thermometers_map_gradient(thermometer_model):
    net = kn.BayesNet(thermometer_model.temperature.getConnectedGraph())
    gradient_optimizer = kn.GradientOptimizer(net)

    logProb = gradient_optimizer.max_a_posteriori()
    assert logProb < 0.

    temperature = thermometer_model.temperature.getValue().scalar()
    assert 20.995 < temperature <  21.005


def test_thermometers_max_likelihood_gradient(thermometer_model):
    net = kn.BayesNet(thermometer_model.temperature.getConnectedGraph())
    gradient_optimizer = kn.GradientOptimizer(net)

    logProb = gradient_optimizer.max_likelihood()
    assert logProb < 0.

    temperature = thermometer_model.temperature.getValue().scalar()
    assert 20.995 < temperature <  21.005