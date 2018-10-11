import keanu as kn
import numpy as np
import pytest


@pytest.fixture
def generic():
    pass

@pytest.mark.parametrize("num, expected_java_class", [
    (1, "ScalarIntegerTensor"),
    (1.3, "ScalarDoubleTensor"),
    (True, "SimpleBooleanTensor")
])
def test_num_passed_to_Tensor_creates_scalar_tensor(num, expected_java_class):
    t = kn.Tensor(num)
    assert_java_class(t, expected_java_class)
    assert t.isScalar()


def test_cannot_pass_generic_to_Tensor(generic):
    with pytest.raises(NotImplementedError) as excinfo:
        kn.Tensor(generic)

    assert str(excinfo.value) == "Generic types in an ndarray are not supported. Was given {}".format(type(generic))


@pytest.mark.parametrize("arr, expected_java_class", [
    ([1, 2], "Nd4jIntegerTensor"),
    ([3.4, 2.], "Nd4jDoubleTensor"),
    ([True, False], "SimpleBooleanTensor")
])
def test_ndarray_passed_to_Tensor_creates_nonscalar_tensor(arr, expected_java_class):
    ndarray = np.array(arr)
    t = kn.Tensor(ndarray)
    assert_java_class(t, expected_java_class)
    assert not t.isScalar()


def test_cannot_pass_generic_ndarray_to_Tensor(generic):
    with pytest.raises(NotImplementedError) as excinfo:
        kn.Tensor(np.array([generic, generic]))

    assert str(excinfo.value) == "Generic types in an ndarray are not supported. Was given {}".format(type(generic))


def test_cannot_pass_empty_ndarray_to_Tensor():
    with pytest.raises(ValueError) as excinfo:
        kn.Tensor(np.array([]))

    assert str(excinfo.value) == "Cannot infer type because the ndarray is empty"


def assert_java_class(java_object_wrapper, java_class_str):
    assert java_object_wrapper.getClass().getSimpleName() == java_class_str
