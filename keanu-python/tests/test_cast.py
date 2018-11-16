from keanu.cast import cast_to_double, cast_to_integer, cast_to_bool
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
