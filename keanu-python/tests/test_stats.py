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


def test_can_get_correct_autocorrelation_when_nonscalar_input() -> None:
    x_list = [
        np.array([[1., 27.2], [10., 3.2]]),
        np.array([[2., 99.4], [5., 4.6]]),
        np.array([[3., 31.5], [10., 7.8]]),
        np.array([[4., 14.3], [5., 2.1]]),
    ]
    expected = np.array([[[1., 0.25, -0.3, -0.45], [1., -0.27679699, -0.32759603, 0.10439302]],
                         [[1., -0.75, 0.5, -0.25], [1., -0.40761833, -0.24778339, 0.15540172]]])
    actual = np.array([[stats.autocorrelation(x_list, (0, 0)),
                        stats.autocorrelation(x_list, (0, 1))],
                       [stats.autocorrelation(x_list, (1, 0)),
                        stats.autocorrelation(x_list, (1, 1))]])
    np.testing.assert_almost_equal(actual, expected)


def test_autocorr_returns_ndarray_of_correct_dtype() -> None:
    with Model() as m:
        m.uniform = Uniform(0, 1000)
    net = m.to_bayes_net()
    samples = sample(net=net, sample_from=net.get_latent_vertices(), draws=10)
    valid_key = list(samples.keys())[0]
    sample_ = samples.get(valid_key)
    print(sample_)
    assert sample_ is not None
    autocorr = stats.autocorrelation(sample_)
    assert type(autocorr) == np.ndarray
    assert autocorr.dtype == sample_[0].dtype


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


def test_autocorrelation_same_for_streaming_as_batch() -> None:
    with Model() as model:
        model.uniform = Uniform(0, 1000)
    net = model.to_bayes_net()
    draws = 15
    set_starting_state(model)
    samples = sample(net=net, sample_from=net.get_latent_vertices(), draws=draws)
    set_starting_state(model)
    iter_samples = generate_samples(net=net, sample_from=net.get_latent_vertices())

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
