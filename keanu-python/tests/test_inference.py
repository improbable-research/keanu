import keanu as kn
import numpy as np
import pytest
from py4j.java_gateway import java_import


def test_build_bayes_net_with_java_list_of_vertices():
    uniform = kn.UniformInt(0, 1)
    connected_graph = uniform.get_connected_graph()

    assert connected_graph.contains(uniform)
    assert connected_graph.size() == 3

    net = kn.BayesNet(connected_graph)
    vertices = net.get_latent_or_observed_vertices()

    assert vertices.contains(uniform)
    assert vertices.size() == 1

def test_build_bayes_net_with_py_list_of_vertices():
    uniform = kn.UniformInt(0, 1)
    net = kn.BayesNet([uniform])
    vertices = net.get_latent_or_observed_vertices()

    assert vertices.contains(uniform)
    assert vertices.size() == 1

def test_cant_build_bayes_net_if_not_java_or_py_list():
    class Something:
        pass
    something = Something()

    assert something is not list
    assert something is not kn.JavaList

    with pytest.raises(ValueError) as excinfo:
        kn.BayesNet(something)

    assert str(excinfo.value) == "Expected a list. Was given {}".format(Something)

def test_can_get_latent_or_observed_vertices():
    uniform = kn.UniformInt(0, 1)
    poisson = kn.Poisson(uniform)

    uniform.observe(0.5)

    assert not poisson.is_observed()
    assert uniform.is_observed()

    net = kn.BayesNet([poisson, uniform])
    vertices = net.get_latent_or_observed_vertices()

    assert vertices.size() == 2
    assert vertices.contains(uniform)
    assert vertices.contains(poisson)

def test_can_get_latent_vertices():
    uniform = kn.UniformInt(0, 1)
    poisson = kn.Poisson(uniform)

    uniform.observe(0.5)

    assert not poisson.is_observed()
    assert uniform.is_observed()

    net = kn.BayesNet([poisson, uniform])
    vertices = net.get_latent_vertices()

    assert vertices.size() == 1
    assert vertices.contains(poisson)

def test_can_get_observed_vertices():
    uniform = kn.UniformInt(0, 1)
    poisson = kn.Poisson(uniform)

    uniform.observe(0.5)

    assert not poisson.is_observed()
    assert uniform.is_observed()

    net = kn.BayesNet([poisson, uniform])
    vertices = net.get_observed_vertices()

    assert vertices.size() == 1
    assert vertices.contains(uniform)

@pytest.fixture
def net():
    gamma = kn.Gamma(1., 1.)
    exp = kn.Exponential(1.)
    cauchy = kn.Cauchy(gamma, exp)

    return kn.BayesNet(cauchy.get_connected_graph())

def test_can_pass_java_list_to_inference_algorithm(net):
    k = kn.KeanuContext().jvm_view()
    java_import(k, "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")
    algorithm = kn.InferenceAlgorithm(k.MetropolisHastings)

    vertices = net.get_latent_vertices()
    network_samples = algorithm.get_posterior_samples(net, vertices, 3)

    assert network_samples.size() == 3

def test_can_pass_py_list_to_inference_algorithm(net):
    k = kn.KeanuContext().jvm_view()
    java_import(k, "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")
    algorithm = kn.InferenceAlgorithm(k.MetropolisHastings)

    gamma = kn.Gamma(1., 1.)
    cauchy = kn.Cauchy(gamma, 1.)
    net = kn.BayesNet([gamma, cauchy])

    network_samples = algorithm.get_posterior_samples(net, [gamma, cauchy], 3)

    assert network_samples.size() == 3

def test_cant_pass_non_list_to_inference_algorithm(net):
    k = kn.KeanuContext().jvm_view()
    java_import(k, "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")
    algorithm = kn.InferenceAlgorithm(k.MetropolisHastings)

    class Something:
        pass
    with pytest.raises(ValueError) as excinfo:
        algorithm.get_posterior_samples(net, Something(), 3)

    assert str(excinfo.value) == "Expected a list. Was given {}".format(Something)


@pytest.mark.parametrize("algorithm, sample_size", [
    (kn.MetropolisHastings, 3),
    (kn.NUTS, 3),
    (kn.Hamiltonian, 3)
])
def test_can_get_posterior_samples(algorithm, sample_size):
    net = kn.BayesNet([kn.Gamma(1., 1.)])
    samples = algorithm().get_posterior_samples(net, net.get_latent_vertices(), 3)

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
    assert vertex_samples.as_list().size() == network_samples.size()

def test_network_samples_get_double_tensor_samples(double_samples):
    vertex, network_samples = double_samples

    vertex_samples = network_samples.get_double_tensor_samples(vertex)
    assert vertex_samples.as_list().size() == network_samples.size()

    averages = vertex_samples.get_averages()
    assert isinstance(averages, np.ndarray)

def test_network_samples_get_integer_tensor_samples(integer_samples):
    vertex, network_samples = integer_samples

    vertex_samples = network_samples.get_integer_tensor_samples(vertex)
    assert vertex_samples.as_list().size() == network_samples.size()
    assert isinstance(vertex_samples.get_scalar_mode(), int)

    averages = vertex_samples.get_averages()
    assert isinstance(averages, np.ndarray)

def test_can_drop_samples(double_samples):
    vertex, network_samples = double_samples
    initial_size = network_samples.size()

    assert network_samples.drop(1).size() == (initial_size - 1)

def test_can_down_sample(double_samples):
    vertex, network_samples = double_samples
    initial_size = network_samples.size()

    assert network_samples.down_sample(2).size() == (initial_size / 2)

def test_can_get_probability(double_samples):
    vertex, network_samples = double_samples

    prob = network_samples.get(vertex).probability(lambda tensor:tensor.scalar() > 0.)
    assert prob == 1.
