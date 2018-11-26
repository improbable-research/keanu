from typing import Union, List

import numpy as np
import pandas as pd
import pytest
from keanu.tensor import Tensor
from keanu.vertex.base import JavaObjectWrapper


@pytest.fixture
def generic():
    pass


@pytest.mark.parametrize("num, expected_java_class", [(1, "ScalarIntegerTensor"),
                                                      (np.array([1])[0], "ScalarIntegerTensor"),
                                                      (1.3, "ScalarDoubleTensor"),
                                                      (np.array([1.3])[0], "ScalarDoubleTensor"),
                                                      (True, "SimpleBooleanTensor"),
                                                      (np.array([True])[0], "SimpleBooleanTensor")])
def test_num_passed_to_Tensor_creates_scalar_tensor(num: Union[int, float, bool, np.ndarray],
                                                    expected_java_class: str) -> None:
    t = Tensor(num)
    assert_java_class(t, expected_java_class)
    assert t.is_scalar()
    assert t.scalar() == num


@pytest.mark.parametrize("data, expected_java_class", [([[1, 2], [3, 4]], "Nd4jIntegerTensor"),
                                                       ([[1., 2.], [3., 4.]], "Nd4jDoubleTensor"),
                                                       ([[True, False], [True, False]], "SimpleBooleanTensor")])
def test_dataframe_passed_to_Tensor_creates_tensor(data: List[List[Union[int, float, bool]]],
                                                   expected_java_class: str) -> None:
    dataframe = pd.DataFrame(columns=['A', 'B'], data=data)
    t = Tensor(dataframe)

    assert_java_class(t, expected_java_class)

    tensor_value = Tensor._to_ndarray(t.unwrap())
    dataframe_value = dataframe.values

    assert np.array_equal(tensor_value, dataframe_value)


@pytest.mark.parametrize("data, expected_java_class", [([1, 2], "Nd4jIntegerTensor"), ([1], "ScalarIntegerTensor"),
                                                       ([1., 2.], "Nd4jDoubleTensor"), ([1.], "ScalarDoubleTensor"),
                                                       ([True, False], "SimpleBooleanTensor"),
                                                       ([True], "SimpleBooleanTensor")])
def test_series_passed_to_Tensor_creates_tensor(data: List[Union[int, float, bool]], expected_java_class: str) -> None:
    series = pd.Series(data)
    t = Tensor(series)

    assert_java_class(t, expected_java_class)

    tensor_value = Tensor._to_ndarray(t.unwrap())
    series_value = series.values

    assert len(tensor_value) == len(series_value)
    assert tensor_value.shape == (len(series_value), 1)
    assert series_value.shape == (len(series_value),)

    assert np.array_equal(tensor_value.flatten(), series_value.flatten())


def test_cannot_pass_generic_to_Tensor(generic) -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        Tensor(generic)

    assert str(excinfo.value) == "Generic types in an ndarray are not supported. Was given {}".format(type(generic))


@pytest.mark.parametrize("arr, expected_java_class", [([1, 2], "Nd4jIntegerTensor"), ([3.4, 2.], "Nd4jDoubleTensor"),
                                                      ([True, False], "SimpleBooleanTensor")])
def test_ndarray_passed_to_Tensor_creates_nonscalar_tensor(arr: Union[int, float, bool],
                                                           expected_java_class: str) -> None:
    ndarray = np.array(arr)
    t = Tensor(ndarray)
    assert_java_class(t, expected_java_class)
    assert not t.is_scalar()


def test_cannot_pass_generic_ndarray_to_Tensor(generic) -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        Tensor(np.array([generic, generic]))

    assert str(excinfo.value) == "Generic types in an ndarray are not supported. Was given {}".format(type(generic))


def test_cannot_pass_empty_ndarray_to_Tensor() -> None:
    with pytest.raises(ValueError) as excinfo:
        Tensor(np.array([]))

    assert str(excinfo.value) == "Cannot infer type because the ndarray is empty"


@pytest.mark.parametrize("value", [(np.array([[1, 2], [3, 4]])), (3)])
def test_convert_java_tensor_to_ndarray(value: np.ndarray) -> None:
    t = Tensor(value)
    ndarray = Tensor._to_ndarray(t.unwrap())

    assert type(ndarray) == np.ndarray
    assert (value == ndarray).all()


def assert_java_class(java_object_wrapper: JavaObjectWrapper, java_class_str: str) -> None:
    assert java_object_wrapper.get_class().getSimpleName() == java_class_str
