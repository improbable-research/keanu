import keanu as kn
import numpy as np
import pytest

def test_integer_passed_to_Tensor_creates_ScalarIntegerTensor():
    t = kn.Tensor(1)
    assert_java_class(t, "ScalarIntegerTensor")
    assert t.isScalar()


def test_double_passed_to_Tensor_creates_ScalarDoubleTensor():
    t = kn.Tensor(1.)
    assert_java_class(t, "ScalarDoubleTensor")
    assert t.isScalar()


def test_bool_passed_to_Tensor_creates_SimpleBooleanTensor():
    t = kn.Tensor(True)
    assert_java_class(t, "SimpleBooleanTensor")
    assert t.isScalar()


def test_cannot_pass_generic_to_Tensor():
    with pytest.raises(ValueError) as excinfo:
        kn.Tensor(GenericExampleClass())

    assert str(excinfo.value) == "Generic types in a tensor are not supported. Was given {}".format(GenericExampleClass)


def test_integer_numpy_tensor_passed_to_Tensor_creates_Nd4jIntegerTensor():
    t = kn.Tensor(np.array([1,2]))
    assert_java_class(t, "Nd4jIntegerTensor")
    assert not t.isScalar()


def test_double_numpy_tensor_passed_to_Tensor_creates_Nd4jDoubleTensor():
    t = kn.Tensor(np.array([1.,2.]))
    assert_java_class(t, "Nd4jDoubleTensor")
    assert not t.isScalar()


def test_bool_numpy_tensor_passed_to_Tensor_creates_SimpleBooleanTensor():
    t = kn.Tensor(np.array([True, False]))
    assert_java_class(t, "SimpleBooleanTensor")
    assert not t.isScalar()


def test_cannot_pass_generic_numpy_tensor_to_Tensor():
    with pytest.raises(ValueError) as excinfo:
        kn.Tensor(np.array([GenericExampleClass(), GenericExampleClass()]))

    assert str(excinfo.value) == "Generic types in a tensor are not supported. Was given {}".format(GenericExampleClass)


def test_cannot_pass_empty_numpy_tensor_to_Tensor():
    with pytest.raises(ValueError) as excinfo:
        kn.Tensor(np.array([]))

    assert str(excinfo.value) == "Cannot infer type because tensor is empty"


def assert_vertex_value_equal_numpy(v, np_tensor):
    nd4j_flat = v.getValue().asFlatArray()
    np_flat = np_tensor.flatten().tolist()

    for i in range(len(np_tensor)):
        assert nd4j_flat[i] == np_flat[i]


def assert_vertex_value_equals_scalar(v, scalar):
    assert v.getValue().scalar() == scalar


def assert_java_class(java_object_wrapper, java_class_str):
    assert java_object_wrapper.getClass().getSimpleName() == java_class_str


class GenericExampleClass:
    pass
