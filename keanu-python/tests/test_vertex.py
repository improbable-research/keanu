import keanu as kn
import numpy as np
import pytest


@pytest.fixture
def jvm_view():
    from py4j.java_gateway import java_import
    jvm_view = kn.KeanuContext().jvm_view()
    java_import(jvm_view, "io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex")
    return jvm_view


def test_can_pass_scalar_to_vertex(jvm_view):
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))
    sample = gaussian.sample()

    assert sample.shape == (1, 1)


def test_can_pass_ndarray_to_vertex(jvm_view):
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (np.array([[0.1, 0.4]]), np.array([[0.4, 0.5]])))
    sample = gaussian.sample()

    assert sample.shape == (1, 2)


def test_can_pass_vertex_to_vertex(jvm_view):
    mu = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (mu, 1.))
    sample = gaussian.sample()

    assert sample.shape == (1, 1)


def test_can_pass_array_to_vertex(jvm_view):
    gaussian = kn.Vertex(jvm_view.GaussianVertex, ([3, 3], 0., 1.))
    sample = gaussian.sample()

    assert sample.shape == (3, 3)


def test_cannot_pass_generic_to_vertex(jvm_view):
    class GenericExampleClass:
        pass

    with pytest.raises(ValueError) as excinfo:
        kn.Vertex(jvm_view.GaussianVertex, (GenericExampleClass(), GenericExampleClass()))

    assert str(excinfo.value) == "Can't parse generic argument. Was given {}".format(GenericExampleClass)


def test_vertex_can_observe_scalar(jvm_view):
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))
    gaussian.observe(4.)

    assert type(gaussian.getValue()) == np.ndarray
    assert gaussian.getValue() == 4.


def test_vertex_can_observe_ndarray(jvm_view):
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))

    ndarray = np.array([[1.,2.]])
    gaussian.observe(ndarray)

    assert type(gaussian.getValue()) == np.ndarray
    assert (gaussian.getValue() == ndarray).all()


def test_int_vertex_value_is_a_numpy_array():
    ndarray = np.array([[1, 2], [3, 4]])
    vertex = kn.Const(ndarray)
    value = vertex.getValue()
    assert type(value) == np.ndarray
    assert value.dtype == np.int64 or value.dtype == np.int32
    assert (value == ndarray).all()

def test_float_vertex_value_is_a_numpy_array():
    ndarray = np.array([[1., 2.], [3., 4.]])
    vertex = kn.Const(ndarray)
    value = vertex.getValue()
    assert type(value) == np.ndarray
    assert value.dtype == np.float64
    assert (value == ndarray).all()

def test_boolean_vertex_value_is_a_numpy_array():
    ndarray = np.array([[True, True], [False, True]])
    vertex = kn.Const(ndarray)
    value = vertex.getValue()
    assert type(value) == np.ndarray
    assert value.dtype == np.bool
    assert (value == ndarray).all()

def test_scalar_vertex_value_is_a_numpy_array():
    scalar = 1.
    vertex = kn.Const(scalar)
    value = vertex.getValue()
    assert type(value) == np.ndarray
    assert value.dtype == np.float64
    assert value.shape == (1, 1)
    assert value == scalar
    assert (value == scalar).all()

def test_vertex_sample_is_a_numpy_array():
    mu = np.array([[1., 2.], [3., 4.]])
    sigma = np.array([[.1, .2], [.3, .4]])
    vertex = kn.Gaussian(mu, sigma)
    value = vertex.sample()
    print(value)
    assert type(value) == np.ndarray
    assert value.dtype == np.float64
    assert value.shape == (2, 2)
