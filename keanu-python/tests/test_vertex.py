import numpy as np
import pandas as pd
import pytest
import math
from keanu.vertex.base import Vertex
from keanu.context import KeanuContext
from keanu.vertex import Gaussian, Const, UniformInt, Bernoulli


@pytest.fixture
def jvm_view():
    from py4j.java_gateway import java_import
    jvm_view = KeanuContext().jvm_view()
    java_import(jvm_view, "io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex")
    return jvm_view


def assert_vertex_value_equals_scalar(vertex, expected_type, scalar):
    vertex_value = vertex.get_value()
    expected_value = np.array([scalar]).astype(expected_type)
    assert vertex_value == expected_value
    assert vertex_value.dtype == expected_type


def assert_vertex_value_equals_ndarray(vertex, expected_type, ndarray):
    vertex_value = vertex.get_value()
    expected_value = ndarray.astype(expected_type)
    assert np.array_equal(vertex_value, expected_value)
    assert vertex_value.dtype == expected_type


def assert_vertex_value_equals_pandas(vertex, expected_type, pandas):
    get_value = vertex.get_value()
    expected_value = pandas.values.astype(expected_type).reshape(get_value.shape)
    assert np.array_equal(get_value, expected_value)
    assert get_value.dtype == expected_type


def test_can_pass_scalar_to_vertex(jvm_view) -> None:
    gaussian = Vertex(jvm_view.GaussianVertex, 0., 1.)
    sample = gaussian.sample()

    assert sample.shape == (1, 1)


def test_can_pass_ndarray_to_vertex(jvm_view) -> None:
    gaussian = Vertex(jvm_view.GaussianVertex, np.array([[0.1, 0.4]]), np.array([[0.4, 0.5]]))
    sample = gaussian.sample()

    assert sample.shape == (1, 2)


def test_can_pass_pandas_to_vertex(jvm_view):
    gaussian = Vertex(jvm_view.GaussianVertex, pd.DataFrame(data=[0.1, 0.4]), pd.Series(data=[0.1, 0.4]))
    sample = gaussian.sample()

    assert sample.shape == (2, 1)


def test_can_pass_vertex_to_vertex(jvm_view) -> None:
    mu = Vertex(jvm_view.GaussianVertex, 0., 1.)
    gaussian = Vertex(jvm_view.GaussianVertex, mu, 1.)
    sample = gaussian.sample()

    assert sample.shape == (1, 1)


def test_can_pass_array_to_vertex(jvm_view) -> None:
    gaussian = Vertex(jvm_view.GaussianVertex, [3, 3], 0., 1.)
    sample = gaussian.sample()

    assert sample.shape == (3, 3)


def test_cannot_pass_generic_to_vertex(jvm_view) -> None:

    class GenericExampleClass:
        pass

    with pytest.raises(ValueError) as excinfo:
        Vertex(jvm_view.GaussianVertex, GenericExampleClass(),
               GenericExampleClass())  # type: ignore # this is expected to fail mypy

    assert str(excinfo.value) == "Can't parse generic argument. Was given {}".format(GenericExampleClass)


def test_you_can_set_and_get_a_value(jvm_view) -> None:
    gaussian = Vertex(jvm_view.GaussianVertex, 0., 1.)
    gaussian.set_value(4.)
    assert gaussian.get_value() == 4.


def test_you_can_cascade_a_value(jvm_view) -> None:
    gaussian1 = Vertex(jvm_view.GaussianVertex, 0., 1.)
    gaussian2 = Vertex(jvm_view.GaussianVertex, 0., 1.)
    sum_of_gaussians = gaussian1 + gaussian2
    gaussian1.set_value(4.)
    gaussian2.set_and_cascade(3.)
    assert sum_of_gaussians.get_value() == 7.


def test_vertex_can_observe_scalar(jvm_view) -> None:
    gaussian = Vertex(jvm_view.GaussianVertex, 0., 1.)
    gaussian.observe(4.)

    assert type(gaussian.get_value()) == np.ndarray
    assert gaussian.get_value() == 4.


def test_vertex_can_observe_ndarray(jvm_view) -> None:
    gaussian = Vertex(jvm_view.GaussianVertex, 0., 1.)

    ndarray = np.array([[1., 2.]])
    gaussian.observe(ndarray)

    assert type(gaussian.get_value()) == np.ndarray
    assert (gaussian.get_value() == ndarray).all()


def test_int_vertex_value_is_a_numpy_array() -> None:
    ndarray = np.array([[1, 2], [3, 4]])
    vertex = Const(ndarray)
    value = vertex.get_value()
    assert type(value) == np.ndarray
    assert value.dtype == np.int64 or value.dtype == np.int32
    assert (value == ndarray).all()


def test_float_vertex_value_is_a_numpy_array() -> None:
    ndarray = np.array([[1., 2.], [3., 4.]])
    vertex = Const(ndarray)
    value = vertex.get_value()
    assert type(value) == np.ndarray
    assert value.dtype == np.float64
    assert (value == ndarray).all()


def test_boolean_vertex_value_is_a_numpy_array() -> None:
    ndarray = np.array([[True, True], [False, True]])
    vertex = Const(ndarray)
    value = vertex.get_value()
    assert type(value) == np.ndarray
    assert value.dtype == np.bool_
    assert (value == ndarray).all()


def test_scalar_vertex_value_is_a_numpy_array() -> None:
    scalar = 1.
    vertex = Const(scalar)
    value = vertex.get_value()
    assert type(value) == np.ndarray
    assert value.dtype == np.float64
    assert value.shape == (1, 1)
    assert value == scalar
    assert (value == scalar).all()


def test_vertex_sample_is_a_numpy_array() -> None:
    mu = np.array([[1., 2.], [3., 4.]])
    sigma = np.array([[.1, .2], [.3, .4]])
    vertex = Gaussian(mu, sigma)
    value = vertex.sample()
    assert type(value) == np.ndarray
    assert value.dtype == np.float64
    assert value.shape == (2, 2)


def test_get_connected_graph(jvm_view) -> None:
    gaussian = Vertex(jvm_view.GaussianVertex, 0., 1.)
    connected_graph = set(gaussian.get_connected_graph())

    assert len(connected_graph) == 3


def test_id_str_of_downstream_vertex_is_higher_than_upstream(jvm_view) -> None:
    hyper_params = Vertex(jvm_view.GaussianVertex, 0., 1.)
    gaussian = Vertex(jvm_view.GaussianVertex, 0., hyper_params)

    hyper_params_id = hyper_params.get_id()
    gaussian_id = gaussian.get_id()

    assert type(hyper_params_id) == tuple
    assert type(gaussian_id) == tuple

    assert hyper_params_id < gaussian_id


def test_construct_vertex_with_java_vertex(jvm_view) -> None:
    java_vertex = Vertex(jvm_view.GaussianVertex, 0., 1.).unwrap()
    python_vertex = Vertex(java_vertex)

    assert tuple(java_vertex.getId().getValue()) == python_vertex.get_id()


def test_java_collections_to_generator(jvm_view) -> None:
    gaussian = Vertex(jvm_view.GaussianVertex, 0., 1.)

    java_collections = gaussian.unwrap().getConnectedGraph()
    python_list = list(Vertex._to_generator(java_collections))

    java_vertex_ids = [Vertex._get_python_id(java_vertex) for java_vertex in java_collections]

    assert java_collections.size() == len(python_list)
    assert all(type(element) == Vertex and element.get_id() in java_vertex_ids for element in python_list)


def test_get_vertex_id(jvm_view) -> None:
    gaussian = Vertex(jvm_view.GaussianVertex, 0., 1.)

    java_id = gaussian.unwrap().getId().getValue()
    python_id = gaussian.get_id()

    assert all(value in python_id for value in java_id)


@pytest.mark.parametrize("vertex, expected_type", [(Gaussian(0., 1.), np.floating), (UniformInt(0, 10), np.integer),
                                                   (Bernoulli(0.5), np.bool_)])
@pytest.mark.parametrize("value, assert_vertex_value_equals",
                         [(4, assert_vertex_value_equals_scalar), (5., assert_vertex_value_equals_scalar),
                          (True, assert_vertex_value_equals_scalar),
                          (np.array([[4]]), assert_vertex_value_equals_ndarray),
                          (np.array([[5.]]), assert_vertex_value_equals_ndarray),
                          (np.array([[True]]), assert_vertex_value_equals_ndarray),
                          (np.array([[1, 2], [3, 4]]), assert_vertex_value_equals_ndarray),
                          (pd.Series(data=[4]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[5.]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[True]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[1, 2, 3]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[1., 2., 3.]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[True, False, False]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[4]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[5.]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[True]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[1, 2, 3]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[1., 2., 3.]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[True, False, False]]), assert_vertex_value_equals_pandas)])
def test_you_can_set_value(vertex, expected_type, value, assert_vertex_value_equals):
    vertex.set_value(value)
    assert_vertex_value_equals(vertex, expected_type, value)


@pytest.mark.parametrize("ctor, args, expected_type", [(Gaussian, (0., 1.), np.floating),
                                                       (UniformInt, (0, 10), np.integer), (Bernoulli,
                                                                                           (0.5,), np.bool_)])
@pytest.mark.parametrize("value, assert_vertex_value_equals",
                         [(4, assert_vertex_value_equals_scalar), (5., assert_vertex_value_equals_scalar),
                          (True, assert_vertex_value_equals_scalar),
                          (np.array([[4]]), assert_vertex_value_equals_ndarray),
                          (np.array([[5.]]), assert_vertex_value_equals_ndarray),
                          (np.array([[True]]), assert_vertex_value_equals_ndarray),
                          (np.array([[1, 2], [3, 4]]), assert_vertex_value_equals_ndarray),
                          (pd.Series(data=[4]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[5.]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[True]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[1, 2, 3]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[1., 2., 3.]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[True, False, False]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[4]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[5.]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[True]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[1, 2, 3]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[1., 2., 3.]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[True, False, False]]), assert_vertex_value_equals_pandas)])
def test_you_can_set_and_cascade(ctor, args, expected_type, value, assert_vertex_value_equals):
    vertex1 = ctor(*args)
    vertex2 = ctor(*args)
    equal_vertex = vertex1 == vertex2
    not_equal_vertex = vertex1 != vertex2

    vertex1.set_value(value)
    vertex2.set_and_cascade(value)

    assert_vertex_value_equals(vertex1, expected_type, value)
    assert_vertex_value_equals(vertex2, expected_type, value)

    two_values_are_equal = equal_vertex.get_value()
    assert two_values_are_equal.dtype == np.bool_
    assert two_values_are_equal.shape == vertex1.get_value().shape == vertex2.get_value().shape
    assert np.all(two_values_are_equal)

    two_values_are_not_equal = not_equal_vertex.get_value()
    assert two_values_are_not_equal.dtype == np.bool_
    assert two_values_are_not_equal.shape == vertex1.get_value().shape == vertex2.get_value().shape
    assert np.all(np.invert(two_values_are_not_equal))


@pytest.mark.parametrize("ctor, args, expected_type", [(Gaussian, (0., 1.), np.floating),
                                                       (UniformInt, (0, 10), np.integer), (Bernoulli,
                                                                                           (0.5,), np.bool_)])
@pytest.mark.parametrize("value, assert_vertex_value_equals",
                         [(4, assert_vertex_value_equals_scalar), (5., assert_vertex_value_equals_scalar),
                          (True, assert_vertex_value_equals_scalar),
                          (np.array([[4]]), assert_vertex_value_equals_ndarray),
                          (np.array([[5.]]), assert_vertex_value_equals_ndarray),
                          (np.array([[True]]), assert_vertex_value_equals_ndarray),
                          (np.array([[1, 2], [3, 4]]), assert_vertex_value_equals_ndarray),
                          (pd.Series(data=[4]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[5.]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[True]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[1, 2, 3]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[1., 2., 3.]), assert_vertex_value_equals_pandas),
                          (pd.Series(data=[True, False, False]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[4]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[5.]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[True]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[1, 2, 3]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[1., 2., 3.]]), assert_vertex_value_equals_pandas),
                          (pd.DataFrame(data=[[True, False, False]]), assert_vertex_value_equals_pandas)])
def test_you_can_observe(ctor, args, expected_type, value, assert_vertex_value_equals):
    vertex = ctor(*args)
    assert not vertex.is_observed()
    vertex.observe(value)
    assert vertex.is_observed()
    assert_vertex_value_equals(vertex, expected_type, value)
