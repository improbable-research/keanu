import keanu as kn
import numpy as np
import pytest

@pytest.mark.parametrize("arr, expected_java_class", [
    ([[1, 2], [3, 4]], "ConstantIntegerVertex"),
    ([[1., 2.], [3., 4.]], "ConstantDoubleVertex"),
    ([[True, False], [False, True]], "ConstantBoolVertex")
])
def test_const_takes_ndarray(arr, expected_java_class):
    ndarray = np.array(arr)
    v = kn.Const(ndarray)

    assert_java_class(v, expected_java_class)
    assert_vertex_value_equal_ndarray(v, ndarray)


@pytest.mark.parametrize("num, expected_java_class", [
    (3, "ConstantIntegerVertex"),
    (3.4, "ConstantDoubleVertex"),
    (True, "ConstantBoolVertex")
])
def test_const_takes_num(num, expected_java_class):
    v = kn.Const(num)

    assert_java_class(v, expected_java_class)
    assert_vertex_value_equals_scalar(v, num)


def test_const_does_not_take_generic_ndarray():
    ndarray = np.array([[GenericExampleClass()]])
    with pytest.raises(NotImplementedError) as excinfo:
        kn.Const(ndarray)

    assert str(excinfo.value) == "Generic types in an ndarray are not supported. Was given {}".format(GenericExampleClass)


def test_const_does_not_take_generic():
    with pytest.raises(NotImplementedError) as excinfo:
        kn.Const(GenericExampleClass())

    assert str(excinfo.value) == "Argument t must be either an ndarray or an instance of numbers.Number. Was given {} instead".format(GenericExampleClass)


def test_const_does_not_take_empty_ndarray():
    ndarray = np.array([])
    with pytest.raises(ValueError) as excinfo:
        kn.Const(ndarray)

    assert str(excinfo.value) == "Cannot infer type because the ndarray is empty"


def test_const_takes_ndarray_of_rank_one():
    ndarray = np.array([1 ,2])
    v = kn.Const(ndarray)

    assert_vertex_value_equal_ndarray(v, ndarray)


def assert_vertex_value_equal_ndarray(v, ndarray):
    nd4j_flat = v.getValue().asFlatArray()
    np_flat = ndarray.flatten().tolist()

    assert len(nd4j_flat) == len(np_flat)

    for i in range(len(ndarray)):
        assert nd4j_flat[i] == np_flat[i]


def assert_vertex_value_equals_scalar(v, scalar):
    assert v.getValue().scalar() == scalar


def assert_java_class(java_object_wrapper, java_class_str):
    assert java_object_wrapper.getClass().getSimpleName() == java_class_str


class GenericExampleClass:
    pass
