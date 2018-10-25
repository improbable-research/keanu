import keanu as kn
import numpy as np
import pytest
from py4j.java_gateway import java_import

def test_construct_bayes_net():
    uniform = kn.UniformInt(0, 1)
    graph = uniform.get_connected_graph()
    vertex_ids = [vertex.get_id() for vertex in graph]

    assert len(vertex_ids) == 3
    assert uniform.get_id() in vertex_ids

    net = kn.BayesNet(graph)
    latent_vertex_ids = [vertex.get_id() for vertex in net.get_latent_vertices()]

    assert len(latent_vertex_ids) == 1
    assert uniform.get_id() in latent_vertex_ids

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
    vertex_ids = [vertex.get_id() for vertex in getattr(net, get_method)()]

    if observed and continuous:
        assert gamma.get_id() in vertex_ids
    if latent and discrete:
        assert poisson.get_id() in vertex_ids
    if latent and continuous:
        assert cauchy.get_id() in vertex_ids

    assert len(vertex_ids) == (observed and continuous) + (latent and discrete) + (latent and continuous)

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
def net():
    gamma = kn.Gamma(1., 1.)
    exp = kn.Exponential(1.)
    cauchy = kn.Cauchy(gamma, exp)

    return kn.BayesNet(cauchy.get_connected_graph())

def test_metropolis(net):
    samples = kn.sample(net=net, sample_from=net.get_latent_vertices(), draws=1)
