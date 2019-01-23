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


def test_then_and_else_must_be_of_the_same_type() -> None:
    with pytest.raises(TypeError) as excinfo:
        If(True, 1, 1.)

    assert str(excinfo.value) == \
           "The \"then\" and \"else\" clauses must be of the same datatype: <class 'int'> vs <class 'float'>"


def test_predicate_must_be_boolean() -> None:
    with pytest.raises(TypeError) as excinfo:
        If(1, 1, 1)

    assert str(excinfo.value) == "Predicate must be boolean: got keanu.vertex.base.Integer"
