import keanu as kn
import numpy as np
import pytest
from py4j.java_gateway import java_import

@pytest.fixture
def net():
    gamma = kn.Gamma(1., 1.)
    exp = kn.Exponential(1.)
    cauchy = kn.Cauchy(gamma, exp)

    return kn.BayesNet(cauchy.get_connected_graph())

@pytest.mark.parametrize("algo", [
    ("metropolis"),
    ("NUTS"),
    ("hamiltonian")
])
def test_sampling_returns_dict_of_list_of_ndarrays_for_vertices_in_sample_from(algo, net):
    draws = 5
    sample_from = list(net.get_latent_vertices())
    vertex_ids = [vertex.get_id() for vertex in sample_from]

    samples = kn.sample(net=net, sample_from=sample_from, algo=algo, draws=draws)

    assert len(samples) == len(vertex_ids)
    assert type(samples) == dict

    for vertex_id, vertex_samples in samples.items():
        assert vertex_id in vertex_ids

        assert len(vertex_samples) == draws
        assert type(vertex_samples) == list
        assert all(type(sample) == np.ndarray for sample in vertex_samples)


def test_dropping_samples(net):
    draws = 10
    drop = 3

    samples = kn.sample(net=net, sample_from=net.get_latent_vertices(), draws=draws, drop=drop)

    expected_num_samples = draws - drop
    assert all(len(vertex_samples) == expected_num_samples for vertex_id, vertex_samples in samples.items())


def test_down_sample_interval(net):
    draws = 10
    down_sample_interval = 2

    samples = kn.sample(net=net, sample_from=net.get_latent_vertices(), draws=draws, down_sample_interval=down_sample_interval)

    expected_num_samples = draws / down_sample_interval
    assert all(len(vertex_samples) == expected_num_samples for vertex_id, vertex_samples in samples.items())


def test_streaming_samples(net):
    draws = 10
    samples_stream = kn.samples_iter(net=net, sample_from=net.get_latent_vertices(), draws=draws, down_sample_interval=1)
    assert sum(1 for i in samples_stream) == 10

