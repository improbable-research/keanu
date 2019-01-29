from typing import Union

import pytest
import numpy as np
import pandas as pd

from keanu.vartypes import tensor_arg_types
from keanu.vertex import If, Bernoulli, Gaussian, Const, Double, Poisson, Integer, Boolean, Exponential, Vertex


@pytest.mark.parametrize(
    "predicate",
    [True, np.array([True, False]),
     pd.Series([True, False]),
     Bernoulli(0.5),
     Const(np.array([True, False]))])
@pytest.mark.parametrize(
    "data",
    [1., np.array([1., 2.]), pd.Series([1., 2.]),
     Exponential(1.), Const(np.array([1., 2.]))])
def test_you_can_create_a_double_valued_if(predicate: Union[tensor_arg_types, Vertex],
                                           data: Union[tensor_arg_types, Vertex]) -> None:
    thn = data
    els = data
    result = If(predicate, thn, els)
    assert type(result) == Double
    assert result.unwrap().getClass().getSimpleName() == "DoubleIfVertex"


@pytest.mark.parametrize(
    "predicate",
    [True, np.array([True, False]),
     pd.Series([True, False]),
     Bernoulli(0.5),
     Const(np.array([True, False]))])
@pytest.mark.parametrize("data", [1, np.array([1, 2]), pd.Series([1, 2]), Poisson(1), Const(np.array([1, 2]))])
def test_you_can_create_an_integer_valued_if(predicate: Union[tensor_arg_types, Vertex],
                                             data: Union[tensor_arg_types, Vertex]) -> None:
    thn = data
    els = data
    result = If(predicate, thn, els)
    assert type(result) == Integer
    assert result.unwrap().getClass().getSimpleName() == "IntegerIfVertex"


@pytest.mark.parametrize(
    "predicate",
    [True, np.array([True, False]),
     pd.Series([True, False]),
     Bernoulli(0.5),
     Const(np.array([True, False]))])
@pytest.mark.parametrize(
    "data",
    [True, np.array([True, False]),
     pd.Series([True, False]),
     Bernoulli(True),
     Const(np.array([True, False]))])
def test_you_can_create_a_boolean_valued_if(predicate: Union[tensor_arg_types, Vertex],
                                            data: Union[tensor_arg_types, Vertex]) -> None:
    thn = data
    els = data
    result = If(predicate, thn, els)
    assert type(result) == Boolean
    assert result.unwrap().getClass().getSimpleName() == "BooleanIfVertex"


@pytest.mark.parametrize(["thn", "els"], [
    (1, 1.),
    (1., 1),
    (1., True),
    (True, 1.),
])
def test_if_thn_or_els_is_not_float_it_gets_coerced(thn, els) -> None:
    result = If(True, thn, els)
    assert type(result) == Double
    assert result.unwrap().getClass().getSimpleName() == "DoubleIfVertex"
    assert result.sample() == 1.


@pytest.mark.parametrize(["thn", "els"], [
    (1, True),
    (True, 1),
])
def test_if_thn_or_els_is_not_int_it_gets_coerced(thn, els) -> None:
    result = If(True, thn, els)
    assert type(result) == Integer
    assert result.unwrap().getClass().getSimpleName() == "IntegerIfVertex"
    assert result.sample() == 1


@pytest.mark.parametrize("pred", [1, 1., 1.1])
def test_if_predicate_is_not_bool_it_gets_coerced(pred) -> None:
    result = If(pred, 1, 0)
    assert type(result) == Integer
    assert result.unwrap().getClass().getSimpleName() == "IntegerIfVertex"
    assert result.sample() == 1
