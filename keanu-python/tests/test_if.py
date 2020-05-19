from typing import Union

import pytest
import numpy as np
import pandas as pd

from keanu.vartypes import tensor_arg_types
from keanu.vertex import Bernoulli, If, Gaussian, Const, Double, Poisson, Integer, Boolean, Exponential, Vertex, Uniform


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
    assert result.unwrap().getClass().getSimpleName() == "WhereVertex"


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
    assert result.unwrap().getClass().getSimpleName() == "WhereVertex"


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
    assert result.unwrap().getClass().getSimpleName() == "WhereVertex"


@pytest.mark.parametrize(["thn", "els"], [
    (1, 1.),
    (1., 1),
    (1., True),
    (True, 1.),
])
def test_if_thn_or_els_is_not_float_it_gets_coerced(thn, els) -> None:
    result = If(True, thn, els)
    assert type(result) == Double
    assert result.unwrap().getClass().getSimpleName() == "WhereVertex"
    assert result.get_value() == 1.


@pytest.mark.parametrize(["thn", "els"], [
    (1, True),
    (True, 1),
])
def test_if_thn_or_els_is_not_int_it_gets_coerced(thn, els) -> None:
    result = If(True, thn, els)
    assert type(result) == Integer
    assert result.unwrap().getClass().getSimpleName() == "WhereVertex"
    assert result.get_value() == 1


@pytest.mark.parametrize("pred", [1, 1., 1.1])
def test_if_predicate_is_not_bool_it_gets_coerced(pred) -> None:
    result = If(pred, 1, 0)
    assert type(result) == Integer
    assert result.unwrap().getClass().getSimpleName() == "WhereVertex"
    assert result.get_value() == 1


def test_you_get_a_useful_error_message_when_you_use_a_boolean_vertex_in_a_python_if_clause() -> None:
    with pytest.raises(
            TypeError,
            match=
            'Keanu vertices cannot be used as a predicate in a Python "if" statement. Please use keanu.vertex.If instead.'
    ):
        if Uniform(0, 1) == 100:
            pass
