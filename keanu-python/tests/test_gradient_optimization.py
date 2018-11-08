import pytest
from examples import thermometers
from py4j.protocol import Py4JJavaError
from keanu import KeanuRandom, BayesNet, Model
from keanu.algorithm import GradientOptimizer

@pytest.fixture
def model() -> Model:
    KeanuRandom().set_default_random_seed(1)
    model = thermometers.model()
    model.thermometer_one.observe(22.0)
    model.thermometer_two.observe(20.0)
    return model


def test_gradient_op_bayes_net(model : Model) -> None:
    net = BayesNet(model.temperature.get_connected_graph())
    gradient_optimizer = GradientOptimizer(net)
    assert gradient_optimizer.net is net


def test_gradient_op_vertex(model : Model) -> None:
    gradient_optimizer = GradientOptimizer(model.temperature)
    assert len(list(gradient_optimizer.net.get_latent_vertices())) == 1


def test_gradient_op_throws_with_invalid_net_param():
    with pytest.raises(TypeError) as excinfo:
        GradientOptimizer(500)

    assert str(excinfo.value) == "net must be a Vertex or a BayesNet. Was given {}".format(int)


def test_gradient_can_set_max_eval_builder_properties(model : Model) -> None:
    gradient_optimizer = GradientOptimizer(model.temperature, max_evaluations=5)

    with pytest.raises(Py4JJavaError):
        #This throws a Gradient Optimizer: "Reached Max Evaluations" error
        logProb = gradient_optimizer.max_a_posteriori()


def test_thermometers_map_gradient(model : Model) -> None:
    net = BayesNet(model.temperature.get_connected_graph())
    gradient_optimizer = GradientOptimizer(net)
    logProb = gradient_optimizer.max_a_posteriori()
    assert logProb < 0.

    temperature = model.temperature.get_value()
    assert 20.995 < temperature <  21.005


def test_thermometers_max_likelihood_gradient(model : Model) -> None:
    net = BayesNet(model.temperature.get_connected_graph())
    gradient_optimizer = GradientOptimizer(net)
    logProb = gradient_optimizer.max_likelihood()
    assert logProb < 0.

    temperature = model.temperature.get_value()
    assert 20.995 < temperature <  21.005
