from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
import pytest


@pytest.fixture
def java_list_wrapper():

    class JavaListWrapper(JavaObjectWrapper):

        def __init__(self, numbers):
            lst = KeanuContext()._gateway.jvm.java.util.ArrayList()
            for number in numbers:
                lst.add(number)

            super(JavaListWrapper, self).__init__(lst)

        def index_of(self, value):
            return 100

    return JavaListWrapper([1, 2, 3])


def test_java_object_wrapper_cant_call_java_api_with_no_python_impl_if_camel_case(java_list_wrapper) -> None:
    with pytest.raises(AttributeError) as excinfo:
        java_list_wrapper.isEmpty()

    assert str(
        excinfo.value
    ) == "{} has no attribute isEmpty. Make sure the attribute is in snake case. Did you mean is_empty?".format(
        type(java_list_wrapper))


def test_java_object_wrapper_cant_call_java_api_with_python_impl_if_camel_case(java_list_wrapper) -> None:
    with pytest.raises(AttributeError) as excinfo:
        java_list_wrapper.indexOf(1)

    assert str(
        excinfo.value
    ) == "{} has no attribute indexOf. Make sure the attribute is in snake case. Did you mean index_of?".format(
        type(java_list_wrapper))


def test_java_object_wrapper_can_call_java_api_with_no_python_impl_if_snake_case(java_list_wrapper) -> None:
    assert not java_list_wrapper.is_empty()


def test_java_object_wrapper_can_call_java_api_with_no_python_impl_if_both_camel_case_and_snake_case(
        java_list_wrapper) -> None:
    assert java_list_wrapper.get(0) == 1


def test_java_object_wrapper_can_call_python_api(java_list_wrapper) -> None:
    assert java_list_wrapper.index_of(1) == 100


def test_java_object_wrapper_throws_type_error_if_not_unwrapped_and_passed_to_java_object(java_list_wrapper) -> None:
    with pytest.raises(
            TypeError,
            match="Trying to pass {} to a method that expects a JavaObject - did you forget to call unwrap()?".format(
                type(java_list_wrapper))):
        lst = KeanuContext()._gateway.jvm.java.util.ArrayList(java_list_wrapper)
