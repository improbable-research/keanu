from itertools import islice

import numpy as np
import pandas as pd
import pytest

from examples import thermometers
from keanu import BayesNet, KeanuRandom, Model
from keanu.algorithm import sample, generate_samples
from keanu.vertex import Gamma, Exponential, Cauchy, Bernoulli
from typing import Any
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt


@pytest.fixture
def net() -> BayesNet:
    with Model() as m:
        m.gamma = Gamma(1., 1.)
        m.exp = Exponential(1.)
        m.cauchy = Cauchy(m.gamma, m.exp)

    return m.to_bayes_net()


@pytest.mark.parametrize("algo", [("metropolis"), ("NUTS"), ("hamiltonian")])
def test_sampling_returns_dict_of_list_of_ndarrays_for_vertices_in_sample_from(algo: str, net: BayesNet) -> None:
    draws = 5
    sample_from = list(net.get_latent_vertices())
    vertex_labels = [vertex.get_label().getQualifiedName() for vertex in sample_from]

    samples = sample(net=net, sample_from=sample_from, algo=algo, draws=draws)
    assert len(samples) == len(sample_from)
    assert type(samples) == dict

    for label, vertex_samples in samples.items():
        assert label in vertex_labels

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
    assert all(len(vertex_samples) == expected_num_samples for label, vertex_samples in samples.items())


def test_down_sample_interval(net: BayesNet) -> None:
    draws = 10
    down_sample_interval = 2

    samples = sample(
        net=net, sample_from=net.get_latent_vertices(), draws=draws, down_sample_interval=down_sample_interval)

    expected_num_samples = draws / down_sample_interval
    assert all(len(vertex_samples) == expected_num_samples for label, vertex_samples in samples.items())


@pytest.mark.mpl_image_compare(filename='test_sample_with_plot.png', tolerance=20)
def test_sample_with_plot() -> Any:
    with Model() as m:
        m.exp = Exponential(np.ones((2, 2)))
        m.exp.observe(np.array([[3., 2.], [0., 1.]]))
        m.bernoulli = Bernoulli(m.exp)
    net = m.to_bayes_net()

    fig, ax = plt.subplots(1, 1, squeeze=False)
    sample(net=net, sample_from=net.get_observed_vertices(), draws=5, plot=True, ax=ax)
    return fig


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

    for vertex in samples_dataframe:
        np.testing.assert_almost_equal(samples_dataframe[vertex].mean(), np.average(samples[vertex]))


@pytest.mark.mpl_image_compare(filename='test_iter_with_live_plot.png', tolerance=20)
def test_iter_with_live_plot() -> Any:
    with Model() as m:
        m.exp = Exponential(np.ones((2, 2)))
        m.exp.observe(np.array([[3., 2.], [0., 1.]]))
        m.bernoulli = Bernoulli(m.exp)
    net = m.to_bayes_net()

    fig, ax = plt.subplots(1, 1, squeeze=False)
    samples = generate_samples(net=net, sample_from=net.get_observed_vertices(), live_plot=True, refresh_every=5, ax=ax)

    for sample in islice(samples, 5):
        pass

    return fig


def set_starting_state(model: Model) -> None:
    KeanuRandom.set_default_random_seed(1)
    model.temperature.set_value(model.temperature.sample())
    model.thermometer_one.set_value(model.thermometer_one.sample())
    model.thermometer_two.set_value(model.thermometer_two.sample())
