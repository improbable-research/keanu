from itertools import islice

import numpy as np
import pandas as pd
import pytest

from examples import thermometers
from keanu import BayesNet, KeanuRandom, Model
from keanu.algorithm import sample, generate_samples
from keanu.vertex import Gamma, Exponential, Cauchy


@pytest.fixture
def net() -> BayesNet:
    gamma = Gamma(1., 1.)
    exp = Exponential(1.)
    cauchy = Cauchy(gamma, exp)

    return BayesNet(cauchy.get_connected_graph())


@pytest.mark.parametrize("algo", [("metropolis"), ("NUTS"), ("hamiltonian")])
def test_sampling_returns_dict_of_list_of_ndarrays_for_vertices_in_sample_from(algo: str, net: BayesNet) -> None:
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
        assert all(sample.dtype == float for sample in vertex_samples)
        assert all(sample.shape == () for sample in vertex_samples)


def test_dropping_samples(net: BayesNet) -> None:
    draws = 10
    drop = 3

    samples = sample(net=net, sample_from=net.get_latent_vertices(), draws=draws, drop=drop)

    expected_num_samples = draws - drop
    assert all(len(vertex_samples) == expected_num_samples for vertex_id, vertex_samples in samples.items())


def test_down_sample_interval(net: BayesNet) -> None:
    draws = 10
    down_sample_interval = 2

    samples = sample(
        net=net, sample_from=net.get_latent_vertices(), draws=draws, down_sample_interval=down_sample_interval)

    expected_num_samples = draws / down_sample_interval
    assert all(len(vertex_samples) == expected_num_samples for vertex_id, vertex_samples in samples.items())


@pytest.mark.parametrize("algo", [("metropolis"), ("hamiltonian")])
def test_can_iter_through_samples(algo: str, net: BayesNet) -> None:
    draws = 10
    samples = generate_samples(net=net, sample_from=net.get_latent_vertices(), algo=algo, down_sample_interval=1)
    count = 0
    for sample in islice(samples, draws):
        count += 1
    assert count == draws


@pytest.mark.parametrize("algo", [("metropolis"), ("hamiltonian")])
def test_iter_returns_same_result_as_sample(algo: str) -> None:
    draws = 100
    model = thermometers.model()
    net = BayesNet(model.temperature.get_connected_graph())
    set_starting_state(model)
    samples = sample(net=net, sample_from=net.get_latent_vertices(), algo=algo, draws=draws)
    set_starting_state(model)
    iter_samples = generate_samples(net=net, sample_from=net.get_latent_vertices(), algo=algo)

    samples_dataframe = pd.DataFrame()
    [samples_dataframe.append(pd.DataFrame(list(next_sample.items()))) for next_sample in islice(iter_samples, draws)]

    for vertex_id in samples_dataframe:
        np.testing.assert_almost_equal(samples_dataframe[vertex_id].mean(), np.average(samples[vertex_id]))


def set_starting_state(model: Model) -> None:
    KeanuRandom.set_default_random_seed(1)
    model.temperature.set_value(model.temperature.sample())
    model.thermometer_one.set_value(model.thermometer_one.sample())
    model.thermometer_two.set_value(model.thermometer_two.sample())
