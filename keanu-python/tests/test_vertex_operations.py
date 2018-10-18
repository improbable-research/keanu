import keanu as kn
import numpy as np
import pytest
from tests.keanu_assert import tensors_equal


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
    assert tensors_equal(result.getValue(), expected_result)


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
    assert tensors_equal(result.getValue(), expected_result)


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
    assert tensors_equal(result.getValue(), expected_result)

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
    assert tensors_equal(result.getValue(), expected_result)

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
    assert tensors_equal(result.getValue(), expected_result)
