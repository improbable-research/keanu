from typing import List, Union

import numpy as np
import pandas as pd
import pytest

from keanu.vartypes import numpy_types, primitive_types
from keanu.vertex import Const
from keanu.vertex.base import JavaObjectWrapper


@pytest.fixture
def generic():
    pass


@pytest.mark.parametrize("arr, expected_java_class", [([[1, 2], [3, 4]], "ConstantIntegerVertex"),
                                                      ([[1., 2.], [3., 4.]], "ConstantDoubleVertex"),
                                                      ([[True, False], [False, True]], "ConstantBooleanVertex")])
def test_const_takes_ndarray(arr: List[List[primitive_types]], expected_java_class: str) -> None:
    ndarray = np.array(arr)
    v = Const(ndarray)

    assert_java_class(v, expected_java_class)
    assert np.array_equal(v.get_value(), ndarray)


@pytest.mark.parametrize("data, expected_java_class", [([1, 2], "ConstantIntegerVertex"),
                                                       ([1., 2.], "ConstantDoubleVertex"),
                                                       ([True, False], "ConstantBooleanVertex")])
def test_const_takes_panda_series(data: List[primitive_types], expected_java_class: str) -> None:
    series = pd.Series(data)
    v = Const(series)

    assert_java_class(v, expected_java_class)

    vertex_value = v.get_value()
    series_value = series.values

    assert len(vertex_value) == len(series_value)
    assert vertex_value.shape == (2,)
    assert series_value.shape == (2,)

    assert np.array_equal(vertex_value.flatten(), series_value.flatten())


@pytest.mark.parametrize("data, expected_java_class", [([[1, 2], [3, 4]], "ConstantIntegerVertex"),
                                                       ([[1., 2.], [3., 4.]], "ConstantDoubleVertex"),
                                                       ([[True, False], [True, False]], "ConstantBooleanVertex")])
def test_const_takes_panda_dataframe(data: List[List[primitive_types]], expected_java_class: str) -> None:
    dataframe = pd.DataFrame(columns=['A', 'B'], data=data)
    v = Const(dataframe)

    assert_java_class(v, expected_java_class)

    vertex_value = v.get_value()
    dataframe_value = dataframe.values

    assert np.array_equal(vertex_value, dataframe_value)


@pytest.mark.parametrize("num, expected_java_class", [(3, "ConstantIntegerVertex"),
                                                      (np.array([3])[0], "ConstantIntegerVertex"),
                                                      (3.4, "ConstantDoubleVertex"),
                                                      (np.array([3.4])[0], "ConstantDoubleVertex"),
                                                      (True, "ConstantBooleanVertex"),
                                                      (np.array([True])[0], "ConstantBooleanVertex")])
def test_const_takes_num(num: Union[primitive_types, numpy_types], expected_java_class: str) -> None:
    v = Const(num)

    assert_java_class(v, expected_java_class)
    assert v.get_value() == num


def test_const_does_not_take_generic_ndarray(generic) -> None:
    ndarray = np.array([[generic]])
    with pytest.raises(NotImplementedError, match=r"Generic types in an ndarray are not supported. Was given object"):
        Const(ndarray)


def test_const_does_not_take_generic(generic) -> None:
    with pytest.raises(
            NotImplementedError,
            match=r"Argument t must be either an ndarray or an instance of numbers.Number. Was given {} instead".format(
                type(generic))):
        Const(generic)


def test_const_does_not_take_empty_ndarray() -> None:
    ndarray = np.array([])
    with pytest.raises(ValueError, match=r"Cannot infer type because array is empty"):
        Const(ndarray)


def test_const_takes_ndarray_of_rank_one() -> None:
    ndarray = np.array([1, 2])
    v = Const(ndarray)

    assert ndarray.shape == (2,)
    assert v.get_value().shape == (2,)

    assert np.array_equal(v.get_value().flatten(), ndarray.flatten())


def test_const_takes_label() -> None:
    label = "const"
    v = Const(3, label=label)
    assert v.get_label() == label


def test_const_takes_none_label() -> None:
    v = Const(3, label=None)
    assert v.get_label() == None


def assert_java_class(java_object_wrapper: JavaObjectWrapper, java_class_str: str) -> None:
    assert java_object_wrapper.unwrap().getClass().getSimpleName() == java_class_str
