import matplotlib.pyplot as plt

from itertools import islice

import numpy as np
import pandas as pd
import pytest

from examples import thermometers
from keanu import BayesNet, KeanuRandom, Model
from keanu.algorithm import (sample, generate_samples, AcceptanceRateTracker, MetropolisHastingsSampler,
                             HamiltonianSampler, NUTSSampler, PosteriorSamplingAlgorithm)
from keanu.vertex import Gamma, Exponential, Cauchy, KeanuContext, Bernoulli
from typing import Any, Callable


@pytest.fixture
def net() -> BayesNet:
    with Model() as m:
        m.gamma = Gamma(1., 1.)
        m.exp = Exponential(1.)
        m.cauchy = Cauchy(m.gamma, m.exp)

    return m.to_bayes_net()


@pytest.mark.parametrize("algo", [(MetropolisHastingsSampler()), (NUTSSampler()), (HamiltonianSampler())])
def test_sampling_returns_dict_of_list_of_ndarrays_for_vertices_in_sample_from(algo: PosteriorSamplingAlgorithm,
                                                                               net: BayesNet) -> None:
    draws = 5
    sample_from = list(net.get_latent_vertices())
    vertex_labels = [vertex.get_label() for vertex in sample_from]

    samples = sample(net=net, sample_from=sample_from, sampling_algorithm=algo, draws=draws)
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


def test_sample_with_plot(net: BayesNet) -> None:
    KeanuRandom.set_default_random_seed(1)
    _, ax = plt.subplots(3, 1, squeeze=False)
    sample(net=net, sample_from=net.get_latent_vertices(), draws=5, plot=True, ax=ax)

    reorder_subplots(ax)

    assert len(ax) == 3
    assert all(len(ax[i][0].get_lines()) == 1 for i in range(3))
    assert all(len(ax[i][0].get_lines()[0].get_ydata()) == 5 for i in range(3))


def test_can_specify_a_gaussian_proposal_distribution(net: BayesNet) -> None:
    algo = MetropolisHastingsSampler(proposal_distribution="gaussian", proposal_distribution_sigma=np.array(1.))
    generate_samples(net=net, sample_from=net.get_latent_vertices(), sampling_algorithm=algo)


@pytest.mark.parametrize("algo", [(MetropolisHastingsSampler()), (HamiltonianSampler())])
def test_can_iter_through_samples(algo: PosteriorSamplingAlgorithm, net: BayesNet) -> None:
    draws = 10
    samples = generate_samples(
        net=net, sample_from=net.get_latent_vertices(), sampling_algorithm=algo, down_sample_interval=1)
    count = 0
    for sample in islice(samples, draws):
        count += 1
    assert count == draws


@pytest.mark.parametrize("algo", [MetropolisHastingsSampler, HamiltonianSampler])
def test_iter_returns_same_result_as_sample(algo: Callable) -> None:
    draws = 100
    model = thermometers.model()
    net = BayesNet(model.temperature.get_connected_graph())
    set_starting_state(model)
    sampler = algo()
    samples = sample(net=net, sample_from=net.get_latent_vertices(), sampling_algorithm=sampler, draws=draws)
    set_starting_state(model)
    sampler = algo()
    iter_samples = generate_samples(net=net, sample_from=net.get_latent_vertices(), sampling_algorithm=sampler)

    samples_dataframe = pd.DataFrame()
    for iter_sample in islice(iter_samples, draws):
        samples_dataframe = samples_dataframe.append(iter_sample, ignore_index=True)

    for vertex_label in samples_dataframe:
        np.testing.assert_almost_equal(samples_dataframe[vertex_label].mean(), np.average(samples[vertex_label]))


def test_iter_with_live_plot(net: BayesNet) -> None:
    KeanuRandom.set_default_random_seed(1)
    _, ax = plt.subplots(3, 1, squeeze=False)
    samples = generate_samples(net=net, sample_from=net.get_latent_vertices(), live_plot=True, refresh_every=5, ax=ax)

    for sample in islice(samples, 5):
        pass

    reorder_subplots(ax)

    assert len(ax) == 3
    assert all(len(ax[i][0].get_lines()) == 1 for i in range(3))
    assert all(len(ax[i][0].get_lines()[0].get_ydata() == 5) for i in range(3))


def test_can_get_acceptance_rates(net: BayesNet) -> None:
    acceptance_rate_tracker = AcceptanceRateTracker()
    latents = list(net.get_latent_vertices())

    algo = MetropolisHastingsSampler(proposal_distribution='prior', proposal_listeners=[acceptance_rate_tracker])
    samples = sample(net=net, sample_from=latents, sampling_algorithm=algo, drop=3)

    for latent in latents:
        rate = acceptance_rate_tracker.get_acceptance_rate(latent)
        assert 0 <= rate <= 1


def test_can_track_acceptance_rate_when_iterating(net: BayesNet) -> None:
    acceptance_rate_tracker = AcceptanceRateTracker()
    latents = list(net.get_latent_vertices())

    algo = MetropolisHastingsSampler(proposal_distribution='prior', proposal_listeners=[acceptance_rate_tracker])
    samples = generate_samples(net=net, sample_from=latents, sampling_algorithm=algo, drop=3)

    draws = 100
    for _ in islice(samples, draws):
        for latent in latents:
            rate = acceptance_rate_tracker.get_acceptance_rate(latent)
            assert 0 <= rate <= 1


def test_it_throws_if_you_pass_in_a_proposal_listener_but_you_didnt_specify_the_proposal_type(net: BayesNet) -> None:
    with pytest.raises(TypeError) as excinfo:
        algo = MetropolisHastingsSampler(proposal_listeners=[AcceptanceRateTracker()])

    assert str(excinfo.value) == "If you pass in proposal_listeners you must also specify proposal_distribution"


def test_can_specify_nuts_params(net: BayesNet):
    algo = NUTSSampler(1000, 0.65, True, 0.1, 10)

    samples = sample(net, list(net.get_latent_vertices()), algo, draws=500, drop=100)


def set_starting_state(model: Model) -> None:
    KeanuRandom.set_default_random_seed(1)
    model.temperature.set_value(model.temperature.sample())
    model.thermometer_one.set_value(model.thermometer_one.sample())
    model.thermometer_two.set_value(model.thermometer_two.sample())


def reorder_subplots(ax: Any) -> None:
    sorted_titles = [plot[0].get_title() for plot in ax]
    sorted_titles.sort()

    positions = [plot[0].get_position() for plot in ax]

    for plot in ax:
        new_position_index = sorted_titles.index(plot[0].get_title())
        plot[0].set_position(positions[new_position_index])
