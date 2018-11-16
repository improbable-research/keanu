from keanu.cast import cast_to_double, cast_to_integer, cast_to_bool
from keanu.vertex import Gaussian, UniformInt
from keanu.vartypes import tensor_arg_types
import pytest
import numpy as np
import pandas as pd

@pytest.mark.parametrize("uncasted, casted", [
    (1, 1.),
    (True, 1.),
    (np.array([1]), np.array([1.])),
    (np.array([True]), np.array([1.])),
    (pd.DataFrame(data=[1]), np.array([1.])),
    (pd.DataFrame(data=[True]), np.array([1.])),
    (pd.Series(data=[1]), np.array([1.])),
    (pd.Series(data=[True]), np.array([1.])),
])
def test_cast_to_double(uncasted: tensor_arg_types, casted: tensor_arg_types) -> None:
    assert cast_to_double(uncasted) == casted

@pytest.mark.parametrize("uncasted, casted", [
    (1., 1),
    (True, 1),
    (np.array([1.]), np.array([1])),
    (np.array([True]), np.array([1])),
    (pd.DataFrame(data=[1.]), np.array([1])),
    (pd.DataFrame(data=[True]), np.array([1])),
    (pd.Series(data=[1.]), np.array([1])),
    (pd.Series(data=[True]), np.array([1]))
])
def test_cast_to_integer(uncasted: tensor_arg_types, casted: tensor_arg_types) -> None:
    assert cast_to_integer(uncasted) == casted

@pytest.mark.parametrize("uncasted, casted", [
    (1., True),
    (1, True),
    (np.array([1.]), np.array([True])),
    (np.array([1]), np.array([True])),
    (pd.DataFrame(data=[1.]), np.array([True])),
    (pd.DataFrame(data=[1]), np.array([True])),
    (pd.Series(data=[1.]), np.array([True])),
    (pd.Series(data=[1]), np.array([True]))
])
def test_cast_to_bool(uncasted: tensor_arg_types, casted: tensor_arg_types) -> None:
    assert cast_to_bool(uncasted) == casted

@pytest.mark.parametrize("mu, sigma", [
    (0, 1),
    (False, True),
    (np.array([0]), np.array([1])),
    (np.array([False]), np.array([True])),
    (pd.DataFrame(data=[0]), pd.DataFrame(data=[1])),
    (pd.DataFrame(data=[False]), pd.DataFrame(data=[True])),
    (pd.Series(data=[0]), pd.Series(data=[1])),
    (pd.Series(data=[False]), pd.Series(data=[True]))
])
def test_cast_for_gaussian(mu: tensor_arg_types, sigma: tensor_arg_types) -> None:
    gaussian_with_double_params = Gaussian(0., 1.)
    gaussian_with_non_double_params = Gaussian(mu, sigma)

    val = 0.5

    assert gaussian_with_double_params.logprob(val) == gaussian_with_non_double_params.logprob(val)

@pytest.mark.parametrize("min, max", [
    (0., 5.),
    (np.array([0.]), np.array([5.])),
    (pd.DataFrame(data=[0.]), pd.DataFrame(data=[5.])),
    (pd.Series(data=[0]), pd.Series(data=[5.])),
])
def test_cast_for_uniform(min: tensor_arg_types, max: tensor_arg_types) -> None:
    uniform_with_int_params = UniformInt(0, 5)
    uniform_with_non_int_params = UniformInt(min, max)

    val = 1
    assert uniform_with_int_params.logprob(val) == uniform_with_non_int_params.logprob(val)
