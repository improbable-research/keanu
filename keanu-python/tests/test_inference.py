import keanu as kn
import numpy as np
import pytest
from py4j.java_gateway import java_import

def test_construct_bayes_net_with_java_list_of_vertices():
    uniform = kn.UniformInt(0, 1)
    java_list = uniform.get_connected_graph()

    assert java_list.contains(uniform.unwrap())
    assert java_list.size() == 3

    net = kn.BayesNet(java_list)
    vertices = net.get_latent_vertices()

    assert vertices.contains(uniform.unwrap())
    assert vertices.size() == 1

def test_construct_bayes_net_with_python_list_of_vertices():
    uniform = kn.UniformInt(0, 1)
    python_list = [uniform]

    net = kn.BayesNet(python_list)
    vertices = net.get_latent_vertices()

    assert vertices.contains(uniform.unwrap())
    assert vertices.size() == 1

def test_cant_construct_bayes_net_if_not_java_or_python_list():
    class Something:
        pass
    something = Something()

    assert something is not list
    assert something is not kn.JavaList

    with pytest.raises(TypeError) as excinfo:
        kn.BayesNet(something)

    assert str(excinfo.value) == "Expected a list. Was given {}".format(Something)

@pytest.mark.parametrize("get_method, latent, observed, continuous, discrete", [
    ("get_latent_or_observed_vertices", True, True, True, True),
    ("get_latent_vertices", True, False, True, True),
    ("get_observed_vertices", False, True, True, True),
    ("get_continuous_latent_vertices", True, False, True, False),
    ("get_discrete_latent_vertices", True, False, False, True)
])
def test_can_get_vertices_from_bayes_net(get_method, latent, observed, continuous, discrete):
    gamma = kn.Gamma(1., 1.)
    gamma.observe(0.5)

    poisson = kn.Poisson(gamma)
    cauchy = kn.Cauchy(gamma, 1.)

    assert gamma.is_observed()
    assert not poisson.is_observed()
    assert not cauchy.is_observed()

    net = kn.BayesNet([gamma, poisson, cauchy])
    vertices = getattr(net, get_method)()

    if observed and continuous:
        assert vertices.contains(gamma.unwrap())
    if latent and discrete:
        assert vertices.contains(poisson.unwrap())
    if latent and continuous:
        assert vertices.contains(cauchy.unwrap())

    assert vertices.size() == (observed and continuous) + (latent and discrete) + (latent and continuous)

def test_probe_for_non_zero_probability_from_bayes_net():
    gamma = kn.Gamma(1., 1.)
    poisson = kn.Poisson(gamma)

    net = kn.BayesNet([poisson, gamma])

    assert not gamma.has_value()
    assert not poisson.has_value()

    net.probe_for_non_zero_probability(100, kn.KeanuRandom())

    assert gamma.has_value()
    assert poisson.has_value()

@pytest.fixture
def inference_algorithm():
    k = kn.KeanuContext().jvm_view()
    java_import(k, "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")
    return kn.InferenceAlgorithm(k.MetropolisHastings)

@pytest.fixture
def net():
    gamma = kn.Gamma(1., 1.)
    exp = kn.Exponential(1.)
    cauchy = kn.Cauchy(gamma, exp)

    return kn.BayesNet(cauchy.get_connected_graph())

def test_can_pass_java_list_to_inference_algorithm(inference_algorithm, net):
    vertices = net.get_latent_vertices()
    network_samples = inference_algorithm.get_posterior_samples(net, vertices, 3)

    assert network_samples.size() == 3

def test_can_pass_python_list_to_inference_algorithm(inference_algorithm, net):
    gamma = kn.Gamma(1., 1.)
    cauchy = kn.Cauchy(gamma, 1.)
    net = kn.BayesNet([gamma, cauchy])

    network_samples = inference_algorithm.get_posterior_samples(net, [gamma, cauchy], 3)

    assert network_samples.size() == 3

def test_cant_pass_non_list_to_inference_algorithm(inference_algorithm, net):
    class Something:
        pass
    with pytest.raises(TypeError) as excinfo:
        inference_algorithm.get_posterior_samples(net, Something(), 3)

    assert str(excinfo.value) == "Expected a list. Was given {}".format(Something)

@pytest.mark.parametrize("algorithm", [
    (kn.MetropolisHastings),
    (kn.NUTS),
    (kn.Hamiltonian)
])
def test_can_get_posterior_samples(algorithm):
    net = kn.BayesNet([kn.Gamma(1., 1.)])
    sample_size = 1
    samples = algorithm().get_posterior_samples(net, net.get_latent_vertices(), sample_size)

    assert samples.size() == sample_size

@pytest.fixture
def double_samples():
    vertex = kn.Gamma(1., 1.)
    net = kn.BayesNet([vertex])
    return vertex, kn.MetropolisHastings().get_posterior_samples(net, net.get_latent_vertices(), 10)

@pytest.fixture
def integer_samples():
    vertex = kn.UniformInt(0, 10)
    net = kn.BayesNet([vertex])
    return vertex, kn.MetropolisHastings().get_posterior_samples(net, net.get_latent_vertices(), 10)

def test_network_samples_get_vertex_samples(double_samples):
    vertex, network_samples = double_samples

    vertex_samples = network_samples.get(vertex)
    assert vertex_samples.unwrap().asList().size() == network_samples.size()

def test_network_samples_get_double_tensor_samples(double_samples):
    vertex, network_samples = double_samples

    vertex_samples = network_samples.get_double_tensor_samples(vertex)
    assert vertex_samples.unwrap().asList().size() == network_samples.size()

def test_network_samples_get_integer_tensor_samples(integer_samples):
    vertex, network_samples = integer_samples

    vertex_samples = network_samples.get_integer_tensor_samples(vertex)
    assert vertex_samples.unwrap().asList().size() == network_samples.size()

def test_network_samples_can_drop_samples(double_samples):
    vertex, network_samples = double_samples
    initial_size = network_samples.size()

    assert network_samples.drop(1).size() == (initial_size - 1)

def test_network_samples_can_down_sample(double_samples):
    vertex, network_samples = double_samples
    initial_size = network_samples.size()

    assert network_samples.down_sample(2).size() == (initial_size / 2)

def test_network_samples_can_get_probability(double_samples):
    vertex, network_samples = double_samples

    prob = network_samples.get(vertex).probability(lambda tensor:tensor.scalar() > 0.)
    assert prob == 1.

@pytest.fixture
def gamma_vertex_samples():
    vertex = kn.Gamma(1., 1.)

    net = kn.BayesNet([vertex])
    return kn.MetropolisHastings().get_posterior_samples(net, net.get_latent_vertices(), 1).get_double_tensor_samples(vertex)

def test_vertex_samples_get_averaes_returns_ndarray(gamma_vertex_samples):
    samples = gamma_vertex_samples.get_averages()

    assert type(samples) == np.ndarray

def test_vertex_samples_probability_is_greater_than_zero(gamma_vertex_samples):
    prob = gamma_vertex_samples.probability(lambda sample: sample.scalar() > 0)

    assert prob == 1.

def test_get_mode_returns_values_greater_than_zero(gamma_vertex_samples):
    mode = gamma_vertex_samples.get_mode()

    assert type(mode) == np.ndarray
    assert (mode > 0).all()
