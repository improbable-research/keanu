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


def test_can_do_vertex_greater_than_vertex(jvm_view):
    v1 = kn.Const(np.array([10., 20.]))
    v2 = kn.Const(np.array([15., 15.]))
    result = v1 > v2
    assert type(result) == kn.Vertex
    assert tensors_equal(result.getValue(), np.array([[False], [True]]))

def test_can_do_vertex_greater_than_np_array(jvm_view):
    v1 = kn.Const(np.array([10., 20.]))
    v2 = np.array([15., 15.])
    result = v1 > v2
    assert type(result) == kn.Vertex
    assert tensors_equal(result.getValue(), np.array([[False], [True]]))

def test_can_do_vertex_greater_than_number(jvm_view):
    v1 = kn.Const(np.array([10., 20.]))
    v2 = 15.
    result = v1 > v2
    assert type(result) == kn.Vertex
    assert tensors_equal(result.getValue(), np.array([[False], [True]]))


def test_can_do_vertex_plus_vertex(jvm_view):
    v1 = kn.Const(np.array([1., 2.]))
    v2 = kn.Const(np.array([10., 20.]))
    result = v1 + v2
    assert type(result) == kn.Vertex
    assert tensors_equal(result.getValue(), np.array([[11], [22]]))

def test_can_do_vertex_plus_np_array(jvm_view):
    v1 = kn.Const(np.array([1., 2.]))
    v2 = np.array([10., 20.])
    result = v1 + v2
    assert type(result) == kn.Vertex
    assert tensors_equal(result.getValue(), np.array([[11], [22]]))

def test_can_do_vertex_plus_number(jvm_view):
    v1 = kn.Const(np.array([1., 2.]))
    v2 = 10.
    result = v1 + v2
    assert type(result) == kn.Vertex
    print(result.getValue())
    assert tensors_equal(result.getValue(), np.array([[11], [12]]))


def test_can_do_vertex_minus_vertex(jvm_view):
    v1 = kn.Const(np.array([10., 20.]))
    v2 = kn.Const(np.array([1., 2.]))
    result = v1 - v2
    assert type(result) == kn.Vertex
    assert tensors_equal(result.getValue(), np.array([[9], [18]]))

def test_can_do_vertex_minus_np_array(jvm_view):
    v1 = kn.Const(np.array([10., 20.]))
    v2 = np.array([1., 2.])
    result = v1 - v2
    assert type(result) == kn.Vertex
    assert tensors_equal(result.getValue(), np.array([[9], [18]]))

def test_can_do_vertex_minus_number(jvm_view):
    v1 = kn.Const(np.array([10., 20.]))
    v2 = 1.
    result = v1 - v2
    assert type(result) == kn.Vertex
    assert tensors_equal(result.getValue(), np.array([[9], [19]]))


def test_can_do_vertex_times_vertex(jvm_view):
    v1 = kn.Const(np.array([3., 2.]))
    v2 = kn.Const(np.array([5., 7.]))
    result = v1 * v2
    assert type(result) == kn.Vertex
    assert tensors_equal(result.getValue(), np.array([[15], [14]]))

def test_can_do_vertex_times_np_array(jvm_view):
    v1 = kn.Const(np.array([3., 2.]))
    v2 = np.array([5., 7.])
    result = v1 * v2
    assert type(result) == kn.Vertex
    assert tensors_equal(result.getValue(), np.array([[15], [14]]))

def test_can_do_vertex_times_number(jvm_view):
    v1 = kn.Const(np.array([3., 2.]))
    v2 = 5.
    result = v1 * v2
    assert type(result) == kn.Vertex
    assert tensors_equal(result.getValue(), np.array([[15], [10]]))