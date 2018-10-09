import keanu as kn
import numpy as np
import pytest

def test_const_takes_int_numpy_tensor():
    np_tensor = np.array([[1, 2], [3, 4]])
    kn.Const(np_tensor)


def test_const_takes_int():
    kn.Const(3)


def test_const_takes_double_numpy_tensor():
    np_tensor = np.array([[1., 2.], [3., 4.]])
    kn.Const(np_tensor)


def test_const_takes_double():
    kn.Const(3.)


def test_const_takes_bool_numpy_tensor():
    np_tensor = np.array([[True, True], [False, True]])
    tensor = kn.Const(np_tensor)


def test_const_takes_bool():
    kn.Const(True)


class Temp:
    pass


def test_const_does_not_take_class_numpy_tensor():
    np_tensor = np.array([[Temp()]])
    with pytest.raises(ValueError):
        kn.Const(np_tensor)


def test_const_does_not_take_class():
    with pytest.raises(ValueError):
        kn.Const(Temp())


def test_const_does_not_take_empty_numpy_tensor():
    kn.Tensor(3)
    np_tensor = np.array([])
    with pytest.raises(ValueError):
        kn.Const(np_tensor)


def test_const_takes_numpy_tensor_of_rank_one():
    np_tensor = np.array([1 ,2])
    kn.Const(np_tensor)
