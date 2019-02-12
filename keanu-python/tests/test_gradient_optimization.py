import pytest
from py4j.protocol import Py4JJavaError

from examples import thermometers
from keanu import KeanuRandom, BayesNet, Model
from keanu.algorithm import GradientOptimizer, ConjugateGradient, Adam


@pytest.fixture
def model() -> Model:
    KeanuRandom.set_default_random_seed(1)
    model = thermometers.model()

    model.thermometer_one.observe(22.0)
    model.thermometer_two.observe(20.0)
    return model


def test_gradient_op_bayes_net(model: Model) -> None:
    net = BayesNet(model.temperature.get_connected_graph())
    gradient_optimizer = GradientOptimizer(net)
    assert gradient_optimizer.net is net


def test_gradient_op_vertex(model: Model) -> None:
    gradient_optimizer = GradientOptimizer(model.temperature)
    assert len(list(gradient_optimizer.net.get_latent_vertices())) == 1


def test_gradient_op_throws_with_invalid_net_param() -> None:
    with pytest.raises(TypeError, match=r"net must be a Vertex or a BayesNet. Was given {}".format(int)):
        GradientOptimizer(500)  # type: ignore # this is expected to fail mypy


def test_gradient_can_set_max_eval_builder_properties_for_conjugate_gradient(model: Model) -> None:
    gradient_optimizer = GradientOptimizer(model.temperature, ConjugateGradient(max_evaluations=5))

    with pytest.raises(Py4JJavaError, match=r"An error occurred while calling o[\d]*.maxAPosteriori."):
        # This throws a Gradient Optimizer: "Reached Max Evaluations" error
        logProb = gradient_optimizer.max_a_posteriori()


def test_thermometers_map_gradient_with_conjugate_gradient(model: Model) -> None:
    thermometers_map_gradient(model, ConjugateGradient())


def test_thermometers_map_gradient_with_adam(model: Model) -> None:
    thermometers_map_gradient(model, Adam())


def thermometers_map_gradient(model: Model, algorithm) -> None:
    net = BayesNet(model.temperature.get_connected_graph())
    gradient_optimizer = GradientOptimizer(net, algorithm)
    result = gradient_optimizer.max_a_posteriori()
    assert result.fitness() < 0.

    temperature = result.value_for(model.temperature)
    assert 20.99 < temperature < 21.01


def test_thermometers_likelihood_gradient_for_conjugate_gradient(model: Model) -> None:
    thermometers_max_likelihood_gradient(model, ConjugateGradient())


def test_thermometers_likelihood_gradient_for_adam(model: Model) -> None:
    thermometers_max_likelihood_gradient(model, Adam())


def thermometers_max_likelihood_gradient(model: Model, algorithm) -> None:
    net = BayesNet(model.temperature.get_connected_graph())
    gradient_optimizer = GradientOptimizer(net, algorithm)
    result = gradient_optimizer.max_likelihood()
    assert result.fitness() < 0.

    temperature = result.value_for(model.temperature)
    assert 20.99 < temperature < 21.01
