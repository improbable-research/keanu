import keanu as kn
import numpy as np
import pytest
import math


### Comparisons

@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (kn.Const(np.array([1., 2.])), kn.Const(np.array([1., -1.])), np.array([[True], [False]])),
    (kn.Const(np.array([1., 2.])),          np.array([1., -1.]) , np.array([[True], [False]])),
    (         np.array([1., 2.]) , kn.Const(np.array([1., -1.])), np.array([[True], [False]])),
    (kn.Const(np.array([1.    ])),                    1.        , np.array([[True]         ])),
    (kn.Const(np.array([    2.])),                    1.        , np.array([        [False]])),
    (                   1.       , kn.Const(np.array([1.     ])), np.array([[True]         ])),
    (                   1.       , kn.Const(np.array([    -1.])), np.array([        [False]])),
])
def test_can_do_equal_to(lhs, rhs, expected_result):
    result = lhs == rhs
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected_result).all()

@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (kn.Const(np.array([1., 2.])), kn.Const(np.array([1., -1.])), np.array([[False], [True]])),
    (kn.Const(np.array([1., 2.])),          np.array([1., -1.]) , np.array([[False], [True]])),
    (         np.array([1., 2.]) , kn.Const(np.array([1., -1.])), np.array([[False], [True]])),
    (kn.Const(np.array([1.    ])),                    1.        , np.array([[False]        ])),
    (kn.Const(np.array([    2.])),                    1.        , np.array([         [True]])),
    (                   1.       , kn.Const(np.array([1.     ])), np.array([[False]        ])),
    (                   1.       , kn.Const(np.array([    -1.])), np.array([         [True]])),    
])
def test_can_do_not_equal_to(lhs, rhs, expected_result):
    result = lhs != rhs
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected_result).all()

@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (kn.Const(np.array([10., 20.])), kn.Const(np.array([15., 15.])), np.array([[False], [True]])),
    (kn.Const(np.array([10., 20.])),          np.array([15., 15.]) , np.array([[False], [True]])),
    (         np.array([10., 20.]) , kn.Const(np.array([15., 15.])), np.array([[False], [True]])),
    (kn.Const(np.array([10., 20.])),                         15.   , np.array([[False], [True]])),
    (                   10.        , kn.Const(np.array([15.,  5.])), np.array([[False], [True]])),
])
def test_can_do_greater_than(lhs, rhs, expected_result):
    result = lhs > rhs
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected_result).all()


@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (kn.Const(np.array([10., 20.])), kn.Const(np.array([15., 15.])), np.array([[True], [False]])),
    (kn.Const(np.array([10., 20.])),          np.array([15., 15.]) , np.array([[True], [False]])),
    (         np.array([10., 20.]) , kn.Const(np.array([15., 15.])), np.array([[True], [False]])),
    (kn.Const(np.array([10., 20.])),                         15.   , np.array([[True], [False]])),
    (                   10.        , kn.Const(np.array([15.,  5.])), np.array([[True], [False]])),
])
def test_can_do_less_than(lhs, rhs, expected_result):
    result = lhs < rhs
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected_result).all()

@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (kn.Const(np.array([10., 15., 20.])), kn.Const(np.array([15., 15., 15.])), np.array([[False], [True], [True]])),
    (kn.Const(np.array([10., 15., 20.])),          np.array([15., 15., 15.]) , np.array([[False], [True], [True]])),
    (         np.array([10., 15., 20.]) , kn.Const(np.array([15., 15., 15.])), np.array([[False], [True], [True]])),
    (kn.Const(np.array([10., 15., 20.])),                              15.   , np.array([[False], [True], [True]])),
    (                   10.             , kn.Const(np.array([15., 10.,  5.])), np.array([[False], [True], [True]])),
])
def test_can_do_greater_than_or_equal_to(lhs, rhs, expected_result):
    result = lhs >= rhs
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected_result).all()

@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (kn.Const(np.array([10., 15., 20.])), kn.Const(np.array([15., 15., 15.])), np.array([[True], [True], [False]])),
    (kn.Const(np.array([10., 15., 20.])),          np.array([15., 15., 15.]) , np.array([[True], [True], [False]])),
    (         np.array([10., 15., 20.]) , kn.Const(np.array([15., 15., 15.])), np.array([[True], [True], [False]])),
    (kn.Const(np.array([10., 15., 20.])),                              15.   , np.array([[True], [True], [False]])),
    (                   10.             , kn.Const(np.array([15., 10.,  5.])), np.array([[True], [True], [False]])),
])
def test_can_do_less_than_or_equal_to(lhs, rhs, expected_result):
    result = lhs <= rhs
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected_result).all()

### Arithmetic

@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (kn.Const(np.array([10., 20.])), kn.Const(np.array([1., 2.])), np.array([[11], [22]])),
    (kn.Const(np.array([10., 20.])),          np.array([1., 2.]) , np.array([[11], [22]])),
    (         np.array([10., 20.]) , kn.Const(np.array([1., 2.])), np.array([[11], [22]])),
    (kn.Const(np.array([10., 20.])),                        2.   , np.array([[12], [22]])),
    (                   10.        , kn.Const(np.array([1., 2.])), np.array([[11], [12]])),
])
def test_can_do_addition(lhs, rhs, expected_result):
    result = lhs + rhs
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected_result).all()

@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (kn.Const(np.array([10., 20.])), kn.Const(np.array([1., 2.])), np.array([[9], [18]])),
    (kn.Const(np.array([10., 20.])),          np.array([1., 2.]) , np.array([[9], [18]])),
    (         np.array([10., 20.]) , kn.Const(np.array([1., 2.])), np.array([[9], [18]])),
    (kn.Const(np.array([10., 20.])),                        2.   , np.array([[8], [18]])),
    (                   10.        , kn.Const(np.array([1., 2.])), np.array([[9], [ 8]])),
])
def test_can_do_subtraction(lhs, rhs, expected_result):
    result = lhs - rhs
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected_result).all()

@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (kn.Const(np.array([3., 2.])), kn.Const(np.array([5., 7.])), np.array([[15], [14]])),
    (kn.Const(np.array([3., 2.])),          np.array([5., 7.]) , np.array([[15], [14]])),
    (         np.array([3., 2.]) , kn.Const(np.array([5., 7.])), np.array([[15], [14]])),
    (kn.Const(np.array([3., 2.])),                    5.       , np.array([[15], [10]])),
    (                   3.,        kn.Const(np.array([5., 7.])), np.array([[15], [21]])),
])
def test_can_do_multiplication(lhs, rhs, expected_result):
    result = lhs * rhs
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected_result).all()

@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (kn.Const(np.array([15., 10.])), kn.Const(np.array([2., 4.])), np.array([[7.5], [2.5 ]])),
    (kn.Const(np.array([15., 10.])),          np.array([2., 4.]) , np.array([[7.5], [2.5 ]])),
    (         np.array([15., 10.]) , kn.Const(np.array([2., 4.])), np.array([[7.5], [2.5 ]])),
    (kn.Const(np.array([15., 10.])),                    2.       , np.array([[7.5], [5.  ]])),
    (                   15.,         kn.Const(np.array([2., 4.])), np.array([[7.5], [3.75]])),
])
def test_can_do_division(lhs, rhs, expected_result):
    result = lhs / rhs
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected_result).all()    

@pytest.mark.parametrize("lhs, rhs, expected_result", [
    (kn.Const(np.array([3., 2.])), kn.Const(np.array([2., 0.5])), np.array([[9], [1.4142135623730951]])),
    (kn.Const(np.array([3., 2.])),          np.array([2., 0.5]) , np.array([[9], [1.4142135623730951]])),
    (         np.array([3., 2.]) , kn.Const(np.array([2., 0.5])), np.array([[9], [1.4142135623730951]])),
    (kn.Const(np.array([3., 2.])),                    2.        , np.array([[9], [4                 ]])),
    (                   3.,        kn.Const(np.array([2., 0.5])), np.array([[9], [1.7320508075688772]])),
])
def test_can_do_pow(lhs, rhs, expected_result):
    result = lhs ** rhs
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected_result).all()

def test_can_do_compound_operations():
    v1 = kn.Const(np.array([
        [2., 3.], 
        [5., 7.]
        ]))
    v2 = np.array([
        [11., 13.], 
        [17., 19.]]
        )
    v3 = 23.

    result = v1 * v2 - v2 / v1 + v3 * v2
    assert (result.get_value() == np.array([[269.5, 333.6666666666667], [472.6, 567.2857142857142]])).all()


### Unary

def test_can_do_abs():
    v = kn.Const(np.array([
        [ 2., -3.], 
        [-5.,  7.]
        ]))

    expected = np.array([
        [ 2.,  3.], 
        [ 5.,  7.]
        ])

    result = abs(v)
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected).all()


def test_can_do_round():
    v = kn.Const(np.array([
        [ 4.4, 4.5,   5.5,  6.6], 
        [-4.4, -4.5, -5.5, -6.6]
        ]))

    expected = np.array([
        [ 4.,  5.,  6.,  7.], 
        [-4., -5., -6., -7.]
        ])

    result = round(v)
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected).all()


def test_can_do_floor():
    v = kn.Const(np.array([
        [ 4.4, 4.5,   5.5,  6.6], 
        [-4.4, -4.5, -5.5, -6.6]
        ]))

    expected = np.array([
        [ 4.,  4.,  5.,  6.], 
        [-5., -5., -6., -7.]
        ])

    result = math.floor(v)
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected).all()


def test_can_do_ceil():
    v = kn.Const(np.array([
        [ 4.4, 4.5,   5.5,  6.6], 
        [-4.4, -4.5, -5.5, -6.6]
        ]))

    expected = np.array([
        [ 5.,  5.,  6.,  7.], 
        [-4., -4., -5., -6.]
        ])

    result = math.ceil(v)
    assert type(result) == kn.Vertex
    assert (result.get_value() == expected).all()        