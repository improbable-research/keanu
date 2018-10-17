import keanu as kn
import numpy as np
import pytest
from tests.keanu_assert import tensors_equal


@pytest.fixture
def jvm_view():
    from py4j.java_gateway import java_import
    jvm_view = kn.KeanuContext().jvm_view()
    java_import(jvm_view, "io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex")
    return jvm_view


def test_can_pass_scalar_to_vertex(jvm_view):
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))
    sample = gaussian.sample()

    assert sample.isScalar()


def test_can_pass_ndarray_to_vertex(jvm_view):
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (np.array([[0.1, 0.4]]), np.array([[0.4, 0.5]])))
    sample = gaussian.sample()

    shape = sample.getShape()
    assert sample.getRank() == 2
    assert shape[0] == 1
    assert shape[1] == 2


def test_can_pass_vertex_to_vertex(jvm_view):
    mu = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (mu, 1.))
    sample = gaussian.sample()

    assert sample.isScalar()


def test_can_pass_array_to_vertex(jvm_view):
    gaussian = kn.Vertex(jvm_view.GaussianVertex, ([3, 3], 0., 1.))
    sample = gaussian.sample()

    shape = sample.getShape()
    assert sample.getRank() == 2
    assert shape[0] == 3
    assert shape[1] == 3


def test_cannot_pass_generic_to_vertex(jvm_view):
    class GenericExampleClass:
        pass

    with pytest.raises(ValueError) as excinfo:
        kn.Vertex(jvm_view.GaussianVertex, (GenericExampleClass(), GenericExampleClass()))

    assert str(excinfo.value) == "Can't parse generic argument. Was given {}".format(GenericExampleClass)


def test_vertex_can_observe_scalar(jvm_view):
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))
    gaussian.observe(4.)

    assert gaussian.getValue().scalar() == 4.


def test_vertex_can_observe_ndarray(jvm_view):
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))

    ndarray = np.array([[1.,2.]])
    gaussian.observe(ndarray)

    nd4j_tensor_flat = gaussian.getValue().asFlatArray()
    assert nd4j_tensor_flat[0] == 1.
    assert nd4j_tensor_flat[1] == 2.


def test_vertex_can_overload_gt(jvm_view):
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))
    sample = gaussian.sample()
    assert sample.isScalar()

    greaterThan = gaussian > np.array([[2., 2.]])

    sample = greaterThan.sample()
    assert not sample.isScalar()

    flat_arr = sample.asFlatArray()
    assert all(isinstance(x, bool) for x in flat_arr)

def test_can_do_addition(jvm_view):
    v1 = kn.Const(np.array([1., 2.]))
    v2 = kn.Const(np.array([10., 20.]))
    sum = v1 + v2
    assert type(sum) == kn.Vertex
    assert tensors_equal(sum.getValue(), np.array([[11], [22]]))

def test_can_do_subtraction(jvm_view):
    v1 = kn.Const(np.array([1., 2.]))
    v2 = kn.Const(np.array([10., 20.]))
    difference = v2 - v1
    assert type(difference) == kn.Vertex
    assert tensors_equal(difference.getValue(), np.array([[9], [18]]))

def test_can_do_multiplication(jvm_view):
    v1 = kn.Const(np.array([3., 2.]))
    v2 = kn.Const(np.array([5., 7.]))
    product = v2 * v1
    assert type(product) == kn.Vertex
    assert tensors_equal(product.getValue(), np.array([[15], [14]]))