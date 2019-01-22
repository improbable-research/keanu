import pytest

from keanu.vertex import IntegerIf, BooleanIf, DoubleIf, Bernoulli, Gaussian, Const, Double, Poisson, Integer, Boolean


@pytest.fixture
def predicate():
    return Bernoulli(0.5)

def test_you_can_create_a_double_valued_if(predicate) -> None:
    thn = Gaussian(0, 1)
    els = Const(10.)
    result = DoubleIf(predicate, thn, els)
    assert type(result) == Double
    assert result.unwrap().getClass().getSimpleName() == "DoubleIfVertex"


def test_you_can_create_an_integer_valued_if(predicate) -> None:
    thn = Poisson(1)
    els = Const(10)
    result = IntegerIf(predicate, thn, els)
    assert type(result) == Integer
    assert result.unwrap().getClass().getSimpleName() == "IntegerIfVertex"


def test_you_can_create_a_boolean_valued_if(predicate) -> None:
    thn = Bernoulli(0.1)
    els = Const(True)
    result = BooleanIf(predicate, thn, els)
    assert type(result) == Boolean
    assert result.unwrap().getClass().getSimpleName() == "BooleanIfVertex"
