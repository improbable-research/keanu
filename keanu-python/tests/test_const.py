import keanu as kn
import numpy as np
import pytest

def test_const_takes_int_numpy_tensor():
    np_tensor = np.array([[1, 2], [3, 4]])
    v = kn.Const(np_tensor)

    assert v.getClass().getSimpleName() == "ConstantIntegerVertex"
    assert_vertex_value_equal_numpy(v, np_tensor)


def test_const_takes_int():
    v = kn.Const(3)

    assert v.getClass().getSimpleName() == "ConstantIntegerVertex"
    assert_vertex_value_equals_scalar(v, 3)

def test_const_takes_double_numpy_tensor():
    np_tensor = np.array([[1., 2.], [3., 4.]])
    v = kn.Const(np_tensor)

    assert v.getClass().getSimpleName() == "ConstantDoubleVertex"
    assert_vertex_value_equal_numpy(v, np_tensor)


def test_const_takes_double():
    v = kn.Const(3.4)

    assert v.getClass().getSimpleName() == "ConstantDoubleVertex"
    assert_vertex_value_equals_scalar(v, 3.4)


def test_const_takes_bool_numpy_tensor():
    np_tensor = np.array([[True, True], [False, True]])
    v = kn.Const(np_tensor)

    assert v.getClass().getSimpleName() == "ConstantBoolVertex"
    assert_vertex_value_equal_numpy(v, np_tensor)


def test_const_takes_bool():
    v = kn.Const(True)

    assert v.getClass().getSimpleName() == "ConstantBoolVertex"
    assert_vertex_value_equals_scalar(v, True)


def test_const_does_not_take_generic_numpy_tensor():
    np_tensor = np.array([[Temp()]])
    with pytest.raises(ValueError) as excinfo:
        kn.Const(np_tensor)

    assert str(excinfo.value) == "Generic types in a tensor are not supported"


def test_const_does_not_take_generic():
    with pytest.raises(ValueError) as excinfo:
        kn.Const(Temp())

    assert str(excinfo.value) == "Argument t must be either a numpy array or an instance of numbers.Number. Was given {} instead".format(Temp)


def test_const_does_not_take_empty_numpy_tensor():
    np_tensor = np.array([])
    with pytest.raises(ValueError) as excinfo:
        kn.Const(np_tensor)

    assert str(excinfo.value) == "Cannot infer type because tensor is empty"


def test_const_takes_numpy_tensor_of_rank_one():
    np_tensor = np.array([1 ,2])
    v = kn.Const(np_tensor)

    assert_vertex_value_equal_numpy(v, np_tensor)


def test_integer_passed_to_Tensor_creates_ScalarIntegerTensor():
    t = kn.Tensor(1)
    assert t.getClass().getSimpleName() == "ScalarIntegerTensor"
    assert t.isScalar()


def test_double_passed_to_Tensor_creates_ScalarDoubleTensor():
    t = kn.Tensor(1.)
    assert t.getClass().getSimpleName() == "ScalarDoubleTensor"
    assert t.isScalar()


def test_bool_passed_to_Tensor_creates_SimpleBooleanTensor():
    t = kn.Tensor(True)
    assert t.getClass().getSimpleName() == "SimpleBooleanTensor"
    assert t.isScalar()


def test_cannot_pass_generic_to_Tensor():
    with pytest.raises(ValueError) as excinfo:
        kn.Tensor(Temp())

    assert str(excinfo.value) == "Generic types in a tensor are not supported"


def test_integer_numpy_tensor_passed_to_Tensor_creates_Nd4jIntegerTensor():
    t = kn.Tensor(np.array([1,2]))
    assert t.getClass().getSimpleName() == "Nd4jIntegerTensor"
    assert not t.isScalar()


def test_double_numpy_tensor_passed_to_Tensor_creates_Nd4jDoubleTensor():
    t = kn.Tensor(np.array([1.,2.]))
    assert t.getClass().getSimpleName() == "Nd4jDoubleTensor"
    assert not t.isScalar()


def test_bool_numpy_tensor_passed_to_Tensor_creates_SimpleBooleanTensor():
    t = kn.Tensor(np.array([True, False]))
    assert t.getClass().getSimpleName() == "SimpleBooleanTensor"
    assert not t.isScalar()


def test_cannot_pass_generic_numpy_tensor_to_Tensor():
    with pytest.raises(ValueError) as excinfo:
        kn.Tensor(np.array([Temp(), Temp()]))

    assert str(excinfo.value) == "Generic types in a tensor are not supported"


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


class Temp:
    pass
