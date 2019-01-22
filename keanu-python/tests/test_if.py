import pytest
import numpy as np
import pandas as pd
from keanu.vertex import IntegerIf, BooleanIf, DoubleIf, Bernoulli, Gaussian, Const, Double, Poisson, Integer, Boolean


@pytest.fixture
def predicate():
    return Bernoulli(0.5)

@pytest.mark.parametrize("data", [1., np.array([1., 2.]), pd.Series([1., 2.])])
def test_you_can_create_a_double_valued_if(data, predicate) -> None:
    thn = Gaussian(data, data)
    els = Const(data)
    result = DoubleIf(predicate, thn, els)
    assert type(result) == Double
    assert result.unwrap().getClass().getSimpleName() == "DoubleIfVertex"


@pytest.mark.parametrize("data", [1, np.array([1, 2]), pd.Series([1, 2])])
def test_you_can_create_an_integer_valued_if(data, predicate) -> None:
    thn = Poisson(data)
    els = Const(data)
    result = IntegerIf(predicate, thn, els)
    assert type(result) == Integer
    assert result.unwrap().getClass().getSimpleName() == "IntegerIfVertex"


@pytest.mark.parametrize("data", [True, np.array([True, False]), pd.Series([True, False])])
def test_you_can_create_a_boolean_valued_if(data, predicate) -> None:
    thn = Const(data)
    els = Const(data)
    result = BooleanIf(predicate, thn, els)
    assert type(result) == Boolean
    assert result.unwrap().getClass().getSimpleName() == "BooleanIfVertex"
