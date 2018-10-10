import keanu as kn
import numpy as np
import pytest
from py4j.java_gateway import java_import

jvm_view = kn.KeanuContext().jvm_view()
java_import(jvm_view, "io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex")


def test_can_pass_scalar_to_vertex():
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))
    sample = gaussian.sample()

    assert sample.isScalar()


def test_can_pass_np_tensor_to_vertex():
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (np.array([[0.1, 0.4]]), np.array([[0.4, 0.5]])))
    sample = gaussian.sample()

    shape = sample.getShape()
    assert sample.getRank() == 2
    assert shape[0] == 1
    assert shape[1] == 2


def test_can_pass_java_object_to_vertex():
    mu = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (mu, 1.))
    sample = gaussian.sample()

    assert sample.isScalar()


def test_can_pass_array_to_vertex():
    gaussian = kn.Vertex(jvm_view.GaussianVertex, ([3, 3], 0., 1.))
    sample = gaussian.sample()

    shape = sample.getShape()
    assert sample.getRank() == 2
    assert shape[0] == 3
    assert shape[1] == 3


def test_cannot_pass_generic_to_vertex():
    class GenericExampleClass:
        pass

    with pytest.raises(ValueError) as excinfo:
        kn.Vertex(jvm_view.GaussianVertex, (GenericExampleClass(), GenericExampleClass()))

    assert str(excinfo.value) == "Can't parse generic argument. Was given {}".format(GenericExampleClass)


def test_vertex_can_observe_scalar():
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))
    gaussian.observe(4.)

    assert gaussian.getValue().scalar() == 4.


def test_vertex_can_observe_tensor():
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))

    np_tensor = np.array([[1.,2.]])
    gaussian.observe(np_tensor)

    nd4j_tensor_flat = gaussian.getValue().asFlatArray()
    assert nd4j_tensor_flat[0] == 1.
    assert nd4j_tensor_flat[1] == 2.


def test_vertex_can_overload_gt():
    gaussian = kn.Vertex(jvm_view.GaussianVertex, (0., 1.))
    sample = gaussian.sample()
    assert sample.isScalar()

    greaterThan = gaussian > np.array([[2., 2.]])

    sample = greaterThan.sample()
    assert not sample.isScalar()

    flat_arr = sample.asFlatArray()
    assert all(isinstance(x, bool) for x in flat_arr)
