from keanu.context import KeanuContext
from keanu.vertex import ConstantInteger, ConstantDouble, ConstantBoolean
import py4j


def test_the_context_is_a_singleton() -> None:
    context1 = KeanuContext()
    context2 = KeanuContext()
    assert context1 == context2


def test_there_is_only_one_jvm_view() -> None:
    view1 = KeanuContext().jvm_view()
    view2 = KeanuContext().jvm_view()
    assert view1 == view2


def test_you_can_convert_a_numpy_array_to_a_java_array() -> None:
    python_list = [1., 2., 3.]
    java_list = KeanuContext().to_java_array(python_list)
    assert type(java_list) == py4j.java_collections.JavaArray
    assert type(java_list[0]) == float
    assert java_list[0] == 1.
    assert java_list[1] == 2.
    assert java_list[2] == 3.

def test_you_can_create_vertex_array_from_different_types() -> None:
    a = ConstantInteger(1)
    b = ConstantDouble(1.)
    c = ConstantBoolean(True)

    context = KeanuContext()
    context.to_java_vertex_array([a, b, c])