import numpy as np
import pytest

from itertools import islice
from keanu import Model, stats, KeanuRandom
from keanu.algorithm import sample, generate_samples
from keanu.vertex import Uniform
import pandas as pd


def test_can_get_correct_autocorrelation() -> None:
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_list = [np.array(a, float) for a in x]
    autocorr = stats.autocorrelation(x_list)
    expected = [1., 0.625, 0.27380952, -0.0297619, -0.26190476, -0.39880952, -0.41666667, -0.29166667]
    np.testing.assert_almost_equal(autocorr, expected)


def test_can_get_autocorrelation_for_samples() -> None:
    with Model() as m:
        m.uniform = Uniform(0, 1000)
    net = m.to_bayes_net()
    samples = sample(net=net, sample_from=net.get_latent_vertices(), algo="metropolis", draws=1000)
    valid_key = list(samples.keys())[0]
    autocorr = stats.autocorrelation(samples.get(valid_key))  # type: ignore


def test_cant_get_autocorrelation_of_np_bools() -> None:
    x = [True, False, False]
    x_list = [np.array(a) for a in x]
    with pytest.raises(ValueError, match="Autocorrelation must be run on a list of numpy floating types."):
        stats.autocorrelation(x_list)


def test_cant_get_autocorrelation_of_np_ints() -> None:
    x = [1, 2, 3]
    x_list = [np.array(a, int) for a in x]
    with pytest.raises(ValueError, match="Autocorrelation must be run on a list of numpy floating types."):
        stats.autocorrelation(x_list)


def test_cant_get_autocorrelation_of_non_scalar_arrays() -> None:
    x_list = [np.array([22.4, 33.3]), np.array([15.4, 11.3])]
    with pytest.raises(ValueError, match="Autocorrelation must be run on a list of single element ndarrays."):
        stats.autocorrelation(x_list)


def test_autocorrelation_same_for_streaming_as_batch() -> None:
    with Model() as model:
        model.uniform = Uniform(0, 1000)
    net = model.to_bayes_net()
    draws = 15
    set_starting_state(model)
    samples = sample(net=net, sample_from=net.get_latent_vertices(), algo="metropolis", draws=draws)
    set_starting_state(model)
    iter_samples = generate_samples(net=net, sample_from=net.get_latent_vertices(), algo="metropolis")

    samples_dataframe = pd.DataFrame()
    for next_sample in islice(iter_samples, draws):
        samples_dataframe = samples_dataframe.append(next_sample, ignore_index=True)

    for vertex_id in samples_dataframe:
        autocorr_streaming = stats.autocorrelation(list(samples_dataframe[vertex_id].values))
        autocorr_batch = stats.autocorrelation(samples[vertex_id])
        np.testing.assert_array_equal(autocorr_batch, autocorr_streaming)


def set_starting_state(model: Model) -> None:
    KeanuRandom.set_default_random_seed(1)
    model.uniform.set_value(model.uniform.sample())
