import numpy as np
import pytest
from keanu.vertex import Gamma, Exponential, Cauchy
from keanu.algorithm import sample
from keanu import BayesNet
from typing import Any

@pytest.fixture
def net() -> BayesNet:
    gamma = Gamma(1., 1.)
    exp = Exponential(1.)
    cauchy = Cauchy(gamma, exp)

    return BayesNet(cauchy.get_connected_graph())

@pytest.mark.parametrize("algo", [
    ("metropolis"),
    ("NUTS"),
    ("hamiltonian")
])
def test_sampling_returns_dict_of_list_of_ndarrays_for_vertices_in_sample_from(algo : str, net : BayesNet) -> None:
    draws = 5
    sample_from = list(net.get_latent_vertices())
    vertex_ids = [vertex.get_id() for vertex in sample_from]

    samples = sample(net=net, sample_from=sample_from, algo=algo, draws=draws)

    assert len(samples) == len(vertex_ids)
    assert type(samples) == dict

    for vertex_id, vertex_samples in samples.items():
        assert vertex_id in vertex_ids

        assert len(vertex_samples) == draws
        assert type(vertex_samples) == list
        assert all(type(sample) == np.ndarray for sample in vertex_samples)


def test_dropping_samples(net : BayesNet) -> None:
    draws = 10
    drop = 3

    samples = sample(net=net, sample_from=net.get_latent_vertices(), draws=draws, drop=drop)

    expected_num_samples = draws - drop
    assert all(len(vertex_samples) == expected_num_samples for vertex_id, vertex_samples in samples.items())


def test_down_sample_interval(net : BayesNet) -> None:
    draws = 10
    down_sample_interval = 2

    samples = sample(net=net, sample_from=net.get_latent_vertices(), draws=draws, down_sample_interval=down_sample_interval)

    expected_num_samples = draws / down_sample_interval
    assert all(len(vertex_samples) == expected_num_samples for vertex_id, vertex_samples in samples.items())
