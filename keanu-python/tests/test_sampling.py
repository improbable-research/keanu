from itertools import islice
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from examples import thermometers
from keanu import BayesNet, KeanuRandom, Model
from keanu.algorithm import (sample, generate_samples, AcceptanceRateTracker, MetropolisHastingsSampler, NUTSSampler,
                             ForwardSampler, PosteriorSamplingAlgorithm)
from keanu.vertex import Gamma, Exponential, Gaussian, Cauchy


@pytest.fixture
def net() -> BayesNet:
    with Model() as m:
        m.gamma = Gamma(1., 1.)
        m.exp = Exponential(1.)
        m.cauchy = Cauchy(m.gamma, m.exp)

    return m.to_bayes_net()


@pytest.fixture
def tensor_net() -> BayesNet:
    with Model() as m:
        m.gamma = Gamma(np.array([1., 1., 1., 1.]).reshape((2, 2)), np.array([2., 2., 2., 2.]).reshape((2, 2)))
        m.exp = Exponential(np.array([1., 1., 1., 1.]).reshape((2, 2)))
        m.add = m.gamma + m.exp

    return m.to_bayes_net()


@pytest.mark.parametrize(
    "algo", [(lambda net: MetropolisHastingsSampler(proposal_distribution="prior", latents=net.iter_latent_vertices())),
             (lambda net: NUTSSampler()), (lambda net: ForwardSampler())])
def test_sampling_returns_dict_of_list_of_ndarrays_for_vertices_in_sample_from(
        algo: Callable[[BayesNet], PosteriorSamplingAlgorithm], net: BayesNet) -> None:
    draws = 5
    sample_from = list(net.iter_latent_vertices())
    samples = sample(net=net, sample_from=sample_from, sampling_algorithm=algo(net), draws=draws)
    assert len(samples) == len(sample_from)
    assert type(samples) == dict
    __assert_valid_samples(draws, samples)


@pytest.mark.parametrize(
    "algo", [(lambda net: MetropolisHastingsSampler(proposal_distribution="prior", latents=net.iter_latent_vertices())),
             (lambda net: NUTSSampler()), (lambda net: ForwardSampler())])
def test_sampling_returns_multi_indexed_dict_of_list_of_scalars_for_tensor_in_sample_from(
        algo: Callable[[BayesNet], PosteriorSamplingAlgorithm], tensor_net: BayesNet) -> None:
    draws = 5
    sample_from = list(tensor_net.iter_latent_vertices())
    samples = sample(net=tensor_net, sample_from=sample_from, sampling_algorithm=algo(tensor_net), draws=draws)
    assert type(samples) == dict
    __assert_valid_samples(draws, samples)


@pytest.mark.parametrize(
    "algo", [(lambda net: MetropolisHastingsSampler(proposal_distribution="prior", latents=net.iter_latent_vertices())),
             (lambda net: NUTSSampler())])
def test_sampling_returns_multi_indexed_dict_of_list_of_scalars_for_mixed_net(
        algo: Callable[[BayesNet], PosteriorSamplingAlgorithm]) -> None:
    exp = Exponential(1.)
    add_rank_2 = exp + np.array([1., 2., 3., 4.]).reshape((2, 2))
    add_rank_3 = exp + np.array([1., 2., 3., 4., 1., 2., 3., 4.]).reshape((2, 2, 2))
    gaussian_rank_2 = Gaussian(add_rank_2, 2.)
    gaussian_rank_3 = Gaussian(add_rank_3, 1.)

    exp.set_label("exp")
    gaussian_rank_2.set_label("gaussian")
    gaussian_rank_3.set_label("gaussian2")

    mixed_net = BayesNet(exp.iter_connected_graph())

    draws = 5
    sample_from = list(mixed_net.iter_latent_vertices())
    vertex_labels = [vertex.get_label() for vertex in sample_from]

    samples = sample(net=mixed_net, sample_from=sample_from, sampling_algorithm=algo(mixed_net), draws=draws)
    assert type(samples) == dict

    __assert_valid_samples(draws, samples)

    assert ('exp', (0,)) in samples
    for i in (0, 1):
        for j in (0, 1):
            assert (('gaussian', (i, j)) in samples)

    df = pd.DataFrame(samples)

    expected_num_columns = {"exp": 1, "gaussian": 4, "gaussian2": 8}

    expected_tuple_size = {"exp": 1, "gaussian": 2, "gaussian2": 3}

    assert len(df.columns.levels[0]) == 3
    for parent_column in df.columns.levels[0]:
        assert parent_column in vertex_labels
        assert len(df[parent_column].columns) == expected_num_columns[parent_column]
        for child_column in df[parent_column].columns:
            assert type(child_column) == tuple
            assert len(child_column) == expected_tuple_size[parent_column]
            assert len(df[parent_column][child_column]) == 5
            assert type(df[parent_column][child_column][0]) == np.float64


def test_sample_dict_can_be_loaded_in_to_dataframe(net: BayesNet) -> None:
    sample_from = list(net.iter_latent_vertices())
    vertex_labels = [vertex.get_label() for vertex in sample_from]

    samples = sample(net=net, sample_from=sample_from, draws=5)
    df = pd.DataFrame(samples)

    for column in df:
        header = df[column].name
        vertex_label = header
        assert vertex_label in vertex_labels
        assert len(df[column]) == 5
        assert type(df[column][0]) == np.float64


def test_multi_indexed_sample_dict_can_be_loaded_in_to_dataframe(tensor_net: BayesNet) -> None:
    sample_from = list(tensor_net.iter_latent_vertices())
    vertex_labels = [vertex.get_label() for vertex in sample_from]

    samples = sample(net=tensor_net, sample_from=sample_from, draws=5)
    df = pd.DataFrame(samples)

    for parent_column in df.columns.levels[0]:
        assert parent_column in vertex_labels

        for child_column in df.columns.levels[1]:
            assert type(child_column) == tuple
            assert len(df[parent_column][child_column]) == 5
            assert type(df[parent_column][child_column][0]) == np.float64


def test_dropping_samples(net: BayesNet) -> None:
    draws = 10
    drop = 3

    samples = sample(net=net, sample_from=net.iter_latent_vertices(), draws=draws, drop=drop)

    expected_num_samples = draws - drop
    assert all(len(vertex_samples) == expected_num_samples for label, vertex_samples in samples.items())


def test_down_sample_interval(net: BayesNet) -> None:
    draws = 10
    down_sample_interval = 2

    samples = sample(
        net=net, sample_from=net.iter_latent_vertices(), draws=draws, down_sample_interval=down_sample_interval)

    expected_num_samples = draws / down_sample_interval
    assert all(len(vertex_samples) == expected_num_samples for label, vertex_samples in samples.items())


def test_sample_with_plot(net: BayesNet) -> None:
    num_plots = 3
    _, ax = plt.subplots(num_plots, 1, squeeze=False)
    sample(net=net, sample_from=net.iter_latent_vertices(), draws=5, plot=True, ax=ax)

    reorder_subplots(ax)

    assert len(ax) == num_plots
    assert all(len(ax[i][0].get_lines()) == 1 for i in range(num_plots))
    assert all(len(ax[i][0].get_lines()[0].get_ydata()) == 5 for i in range(num_plots))


def test_can_specify_a_gaussian_proposal_distribution(net: BayesNet) -> None:
    algo = MetropolisHastingsSampler(
        proposal_distribution="gaussian", latents=net.iter_latent_vertices(), proposal_distribution_sigma=np.array(1.))
    generate_samples(net=net, sample_from=net.iter_latent_vertices(), sampling_algorithm=algo)


@pytest.mark.parametrize(
    "algo",
    [(lambda net: MetropolisHastingsSampler(proposal_distribution='prior', latents=net.iter_latent_vertices()))])
def test_can_iter_through_samples(algo: Callable[[BayesNet], PosteriorSamplingAlgorithm], net: BayesNet) -> None:
    draws = 10
    samples = generate_samples(
        net=net, sample_from=net.iter_latent_vertices(), sampling_algorithm=algo(net), down_sample_interval=1)
    count = 0
    for sample in islice(samples, draws):
        count += 1

    assert count == draws


@pytest.mark.parametrize(
    "algo", [(lambda net: MetropolisHastingsSampler(proposal_distribution="prior", latents=net.iter_latent_vertices())),
             (lambda net: NUTSSampler())])
def test_can_iter_through_tensor_samples(algo: Callable[[BayesNet], PosteriorSamplingAlgorithm],
                                         tensor_net: BayesNet) -> None:
    draws = 10
    samples = generate_samples(
        net=tensor_net,
        sample_from=tensor_net.iter_latent_vertices(),
        sampling_algorithm=algo(tensor_net),
        down_sample_interval=1)
    count = 0
    for sample in islice(samples, draws):
        count += 1
        for distribution in ('exp', 'gamma'):
            for i in (0, 1):
                for j in (0, 1):
                    assert ((distribution, (i, j)) in sample)
    assert count == draws


def test_iter_returns_same_result_as_sample() -> None:
    draws = 100
    model = thermometers.model()
    net = BayesNet(model.temperature.iter_connected_graph())
    set_starting_state(model)
    sampler = MetropolisHastingsSampler(proposal_distribution='prior', latents=net.iter_latent_vertices())
    samples = sample(net=net, sample_from=net.iter_latent_vertices(), sampling_algorithm=sampler, draws=draws)
    set_starting_state(model)
    sampler = MetropolisHastingsSampler(proposal_distribution='prior', latents=net.iter_latent_vertices())
    iter_samples = generate_samples(net=net, sample_from=net.iter_latent_vertices(), sampling_algorithm=sampler)

    samples_dataframe = pd.DataFrame()
    for iter_sample in islice(iter_samples, draws):
        samples_dataframe = samples_dataframe.append(iter_sample, ignore_index=True)

    for vertex_label in samples_dataframe:
        np.testing.assert_almost_equal(samples_dataframe[vertex_label].mean(), np.average(samples[vertex_label]))


def test_iter_with_live_plot(net: BayesNet) -> None:
    num_plots = 3
    _, ax = plt.subplots(num_plots, 1, squeeze=False)
    samples = generate_samples(net=net, sample_from=net.iter_latent_vertices(), live_plot=True, refresh_every=5, ax=ax)

    for sample in islice(samples, 5):
        pass

    reorder_subplots(ax)
    assert len(ax) == num_plots
    assert all(len(ax[i][0].get_lines()) == 1 for i in range(num_plots))
    assert all(len(ax[i][0].get_lines()[0].get_ydata() == 5) for i in range(num_plots))


def test_can_get_acceptance_rates(net: BayesNet) -> None:
    acceptance_rate_tracker = AcceptanceRateTracker()
    latents = list(net.iter_latent_vertices())

    algo = MetropolisHastingsSampler(
        proposal_distribution='prior', latents=net.iter_latent_vertices(), proposal_listeners=[acceptance_rate_tracker])
    samples = sample(net=net, sample_from=latents, sampling_algorithm=algo, drop=3)

    for latent in latents:
        rate = acceptance_rate_tracker.get_acceptance_rate(latent)
        assert 0 <= rate <= 1


def test_can_track_acceptance_rate_when_iterating(net: BayesNet) -> None:
    acceptance_rate_tracker = AcceptanceRateTracker()
    latents = list(net.iter_latent_vertices())

    algo = MetropolisHastingsSampler(
        proposal_distribution='prior', latents=net.iter_latent_vertices(), proposal_listeners=[acceptance_rate_tracker])
    samples = generate_samples(net=net, sample_from=latents, sampling_algorithm=algo, drop=3)

    draws = 100
    for _ in islice(samples, draws):
        for latent in latents:
            rate = acceptance_rate_tracker.get_acceptance_rate(latent)
            assert 0 <= rate <= 1


def test_can_specify_nuts_params(net: BayesNet) -> None:
    algo = NUTSSampler(
        adapt_count=1000,
        target_acceptance_prob=0.65,
        adapt_step_size_enabled=True,
        adapt_potential_enabled=True,
        initial_step_size=0.1,
        max_tree_height=10)

    samples = sample(net, list(net.iter_latent_vertices()), algo, draws=500, drop=100)


def test_sample_throws_if_vertices_in_sample_from_are_missing_labels() -> None:
    sigma = Gamma(1., 1)
    vertex = Gaussian(0., sigma, label="gaussian")

    assert sigma.get_label() is None

    net = BayesNet([sigma, vertex])
    with pytest.raises(ValueError, match=r"Vertices in sample_from must be labelled."):
        samples = sample(net=net, sample_from=net.iter_latent_vertices())


def test_generate_samples_throws_if_vertices_in_sample_from_are_missing_labels() -> None:
    sigma = Gamma(1., 1)
    vertex = Gaussian(0., sigma, label="gaussian")

    assert sigma.get_label() is None

    net = BayesNet([sigma, vertex])
    with pytest.raises(ValueError, match=r"Vertices in sample_from must be labelled."):
        samples = generate_samples(net=net, sample_from=net.iter_latent_vertices())


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


def __assert_valid_samples(draws: int, samples: Dict) -> None:
    for label, vertex_samples in samples.items():
        assert len(vertex_samples) == draws
        assert type(vertex_samples) == list
        assert all(type(sample) == float for sample in vertex_samples)
