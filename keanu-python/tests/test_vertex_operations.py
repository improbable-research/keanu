from typing import SupportsFloat, SupportsRound, Union, Any

import numpy as np
import pytest
import math

from keanu.vartypes import numpy_types
from keanu.vertex import Const
from keanu.vertex.base import Vertex, Double, Integer, Boolean

### Comparisons


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([1., 2.])),    Const(np.array([1., -1.])), np.array([True, False])),
    (Const(np.array([1., 2.])),          np.array([1., -1.]) , np.array([True, False])),
    (Const(np.array([1.    ])),                    1.        , np.array([True       ])),
    (Const(np.array([    2.])),                    1.        , np.array([      False])),
    (Const(np.array([1., 2.])),    Const(np.array([1.     ])), np.array([True, False])),
])
# yapf: enable
def test_can_do_equal_to(lhs: Vertex, rhs: Union[Vertex, numpy_types, float], expected_result: numpy_types) -> None:
    result = lhs == rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()
    result = rhs == lhs  # type: ignore # see https://github.com/python/mypy/issues/5951
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([1., 2.])), Const(np.array([1., -1.])), np.array([False, True])),
    (Const(np.array([1., 2.])),       np.array([1., -1.]) , np.array([False, True])),
    (Const(np.array([1.    ])),                 1.        , np.array([False      ])),
    (Const(np.array([    2.])),                 1.        , np.array([       True])),
    (Const(np.array([1., 2.])), Const(np.array([1.     ])), np.array([False, True])),
])
# yapf: enable
def test_can_do_not_equal_to(lhs: Vertex, rhs: Union[Vertex, numpy_types, float], expected_result: numpy_types) -> None:
    result = lhs != rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()
    result = rhs != lhs  # type: ignore # see https://github.com/python/mypy/issues/5951
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([10., 20.])), Const(np.array([15., 15.])), np.array([False, True])),
    (Const(np.array([10., 20.])),       np.array([15., 15.]) , np.array([False, True])),
    (Const(np.array([10., 20.])),                      15.   , np.array([False, True])),
])
# yapf: enable
def test_can_do_greater_than(lhs: Vertex, rhs: Union[Vertex, numpy_types, float], expected_result: numpy_types) -> None:
    result: Union[Vertex, np._ArrayLike[bool]] = lhs > rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()
    result = rhs > lhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == np.logical_not(expected_result)).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([10., 20.])), Const(np.array([15., 15.])), np.array([True, False])),
    (Const(np.array([10., 20.])),       np.array([15., 15.]) , np.array([True, False])),
    (Const(np.array([10., 20.])),                      15.   , np.array([True, False])),
])
# yapf: enable
def test_can_do_less_than(lhs: Vertex, rhs: Union[Vertex, numpy_types, float], expected_result: numpy_types) -> None:
    result: Union[Vertex, np._ArrayLike[bool]] = lhs < rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()
    result = rhs < lhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == np.logical_not(expected_result)).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([10., 15., 20.])), Const(np.array([15., 15., 15.])), np.array([False, True, True])),
    (Const(np.array([10., 15., 20.])),       np.array([15., 15., 15.]) , np.array([False, True, True])),
    (Const(np.array([10., 15., 20.])),                           15.   , np.array([False, True, True])),
])
# yapf: enable
def test_can_do_greater_than_or_equal_to(lhs: Vertex, rhs: Union[Vertex, numpy_types, float],
                                         expected_result: numpy_types) -> None:
    result = lhs >= rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (np.array([10., 15., 20.]), Const(np.array([15., 15., 15.])), np.array([False, True, True])),
    (10.                      , Const(np.array([15., 10., 5. ])), np.array([False, True, True])),
])
# yapf: enable
def test_can_do_greater_than_or_equal_to_with_vertex_on_rhs(lhs: Union[Vertex, numpy_types, float], rhs: Vertex,
                                                            expected_result: numpy_types) -> None:
    result = lhs >= rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([10., 15., 20.])), Const(np.array([15., 15., 15.])), np.array([True, True, False])),
    (Const(np.array([10., 15., 20.])),       np.array([15., 15., 15.]) , np.array([True, True, False])),
    (Const(np.array([10., 15., 20.])),                           15.   , np.array([True, True, False])),
])
# yapf: enable
def test_can_do_less_than_or_equal_to(lhs: Vertex, rhs: Union[Vertex, numpy_types, float],
                                      expected_result: numpy_types) -> None:
    result = lhs <= rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (np.array([10., 15., 20.]), Const(np.array([15., 15., 15.])), np.array([True, True, False])),
    (10.                      , Const(np.array([15., 10., 5. ])), np.array([True, True, False])),
])
# yapf: enable
def test_can_do_less_than_or_equal_to_with_vertex_on_rhs(lhs: Union[Vertex, numpy_types, float], rhs: Vertex,
                                                         expected_result: numpy_types) -> None:
    result = lhs <= rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


### Arithmetic


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([10., 20.])), Const(np.array([1., 2.])), np.array([11, 22])),
    (Const(np.array([10., 20.])),       np.array([1., 2.]) , np.array([11, 22])),
    (Const(np.array([10., 20.])),                     2.   , np.array([12, 22])),
])
# yapf: enable
def test_can_do_addition(lhs: Union[Vertex], rhs: Union[Vertex, numpy_types, float],
                         expected_result: numpy_types) -> None:
    result: Union[Vertex, np._ArrayLike[Any]] = lhs + rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()
    result = rhs + lhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([10., 20.])), Const(np.array([1., 2.])), np.array([9, 18])),
    (Const(np.array([10., 20.])),       np.array([1., 2.]) , np.array([9, 18])),
    (Const(np.array([10., 20.])),                     2.   , np.array([8, 18])),
])
# yapf: enable
def test_can_do_subtraction(lhs: Vertex, rhs: Union[Vertex, numpy_types, float], expected_result: numpy_types) -> None:
    result: Union[Vertex, np._ArrayLike[Any]] = lhs - rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()
    result = rhs - lhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == -expected_result).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([3., 2.])), Const(np.array([5., 7.])), np.array([15, 14])),
    (Const(np.array([3., 2.])),       np.array([5., 7.]) , np.array([15, 14])),
    (Const(np.array([3., 2.])),                 5.       , np.array([15, 10])),
])
# yapf: enable
def test_can_do_multiplication(lhs: Vertex, rhs: Union[Vertex, numpy_types, float],
                               expected_result: numpy_types) -> None:
    result: Union[Vertex, np._ArrayLike[Any]] = lhs * rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()
    result = rhs * lhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([15., 10.])), Const(np.array([2., 4.])), np.array([7.5, 2.5 ])),
    (Const(np.array([15., 10.])),       np.array([2., 4.]) , np.array([7.5, 2.5 ])),
    (Const(np.array([15., 10.])),                 2.       , np.array([7.5, 5.  ])),
])
# yapf: enable
def test_can_do_division(lhs: Vertex, rhs: Union[Vertex, numpy_types, float], expected_result: numpy_types) -> None:
    result: Union[Vertex, np._ArrayLike[Any]] = lhs / rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()
    result = rhs / lhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == 1. / expected_result).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([15, 10])), Const(np.array([2, 4])), np.array([7, 2])),
    (Const(np.array([15, 10])),       np.array([2, 4]) , np.array([7, 2])),
    (Const(np.array([15, 10])),                 2      , np.array([7, 5])),
])
# yapf: enable
def test_can_do_integer_division(lhs: Vertex, rhs: Union[Vertex, numpy_types, float],
                                 expected_result: numpy_types) -> None:
    result = lhs // rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (np.array([15, 10]), Const(np.array([2, 4])), np.array([7, 2])),
    (15, Const(np.array([2, 4])), np.array([7, 3])),
])
# yapf: enable
def test_can_do_integer_division_with_vertex_on_rhs(lhs: Union[Vertex, numpy_types, float], rhs: Vertex,
                                                    expected_result: numpy_types) -> None:
    result = lhs // rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


# yapf: disable


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (Const(np.array([3., 2.])), Const(np.array([2., 0.5])), np.array([9, 1.4142135623730951])),
    (Const(np.array([3., 2.])),       np.array([2., 0.5]) , np.array([9, 1.4142135623730951])),
    (Const(np.array([3., 2.])),                 2.        , np.array([9, 4                 ])),
])
# yapf: enable
def test_can_do_pow(lhs: Vertex, rhs: Union[Vertex, numpy_types, float], expected_result: numpy_types) -> None:
    result = lhs ** rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


# yapf: disable
@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (np.array([3., 2.]), Const(np.array([2., 0.5])), np.array([9, 1.4142135623730951])),
    (3.                , Const(np.array([2., 0.5])), np.array([9, 1.7320508075688772])),
])
# yapf: enable
def test_can_do_pow_with_vertex_on_rhs(lhs: Union[Vertex, numpy_types, float], rhs: Vertex,
                                       expected_result: numpy_types) -> None:
    result = lhs ** rhs
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected_result).all()


def test_can_do_compound_operations() -> None:
    v1 = Const(np.array([[2., 3.], [5., 7.]]))
    v2 = np.array([[11., 13.], [17., 19.]])
    v3 = 23.

    result = v1 * v2 - v2 / v1 + v3 * v2
    assert (result.get_value() == np.array([[269.5, 333.6666666666667], [472.6, 567.2857142857142]])).all()


### Unary


def test_can_do_abs() -> None:
    v = Const(np.array([[2., -3.], [-5., 7.]]))

    expected = np.array([[2., 3.], [5., 7.]])

    result = abs(v)
    assert isinstance(result, Vertex)
    assert (result.get_value() == expected).all()


def test_can_do_round() -> None:
    v = Const(np.array([[4.4, 4.5, 5.5, 6.6], [-4.4, -4.5, -5.5, -6.6]]))

    expected = np.array([[4., 5., 6., 7.], [-4., -5., -6., -7.]])

    result = round(v)
    assert isinstance(result, Vertex)
    assert (
        result
        .  # type: ignore # see https://stackoverflow.com/questions/53481887/looking-for-a-working-example-of-supportsround
        get_value() == expected).all()


def test_rounding_is_only_supported_to_zero_digits() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        v = Const(1.55)
        round(v, 1)
    assert str(excinfo.value) == "Keanu only supports rounding to 0 digits"


def test_can_do_floor() -> None:
    v = Const(np.array([[4.4, 4.5, 5.5, 6.6], [-4.4, -4.5, -5.5, -6.6]]))

    expected = np.array([[4., 4., 5., 6.], [-5., -5., -6., -7.]])

    result = math.floor(v)
    # assert isinstance(result, Vertex)
    assert (
        result.  # type: ignore # see https://stackoverflow.com/questions/53483596/type-annotations-for-floor-and-ceil
        get_value() == expected).all()


def test_can_do_ceil() -> None:
    v = Const(np.array([[4.4, 4.5, 5.5, 6.6], [-4.4, -4.5, -5.5, -6.6]]))

    expected = np.array([[5., 5., 6., 7.], [-4., -4., -5., -6.]])

    result = math.ceil(v)
    assert isinstance(result, Vertex)
    assert (
        result.  # type: ignore # see https://stackoverflow.com/questions/53483596/type-annotations-for-floor-and-ceil
        get_value() == expected).all()
