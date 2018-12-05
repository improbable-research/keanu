import numpy as np
import pytest

from keanu.vertex import Uniform
from keanu.algorithm import sample
from keanu import Model, stats


def test_can_get_correct_autocorrelation() -> None:
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_list = [np.array(a, float) for a in x]
    autocorr = stats.autocorrelation(x_list)
    expected = [1., 0.625, 0.27380952, -0.0297619, -0.26190476, -0.39880952, -0.41666667, -0.29166667]
    np.testing.assert_almost_equal(autocorr, expected)


def test_can_get_autocorrelation_for_samples() -> None:
    with Model() as m:
        m.gaussian = Uniform(0, 1000)
    net = m.to_bayes_net()
    samples = sample(net=net, sample_from=list(net.get_latent_vertices()), algo="metropolis", draws=1000)
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
