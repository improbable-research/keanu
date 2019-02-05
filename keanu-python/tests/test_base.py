from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
import pytest
from py4j.protocol import Py4JError


@pytest.fixture
def java_list_wrapper():

    class JavaListWrapper(JavaObjectWrapper):

        def __init__(self, numbers):
            lst = KeanuContext()._gateway.jvm.java.util.ArrayList()
            for number in numbers:
                lst.add(number)

            super(JavaListWrapper, self).__init__(lst)

        def get(self, index):
            return self.unwrap().get(index) + 100

    return JavaListWrapper([1, 2, 3])


def test_you_can_call_a_java_method_on_the_unwrapped_object(java_list_wrapper) -> None:
    assert not java_list_wrapper.unwrap().isEmpty()


def test_you_cannot_call_a_java_method_with_snake_case_on_the_unwrapped_object(java_list_wrapper) -> None:
    with pytest.raises(Py4JError, match="Method is_empty\(\[\]\) does not exist"):
        java_list_wrapper.unwrap().is_empty()


def test_you_cannot_call_a_java_method_with_snake_case_on_the_wrapped_object(java_list_wrapper) -> None:
    with pytest.raises(AttributeError, match="{} has no attribute is_empty".format(type(java_list_wrapper))):
        java_list_wrapper.is_empty()


def test_you_can_overload_a_java_method_in_python(java_list_wrapper) -> None:
    assert java_list_wrapper.get(0) == 101


def test_you_cannot_call_a_java_method_that_hasnt_been_overloaded(java_list_wrapper) -> None:
    with pytest.raises(AttributeError, match="{} has no attribute size".format(type(java_list_wrapper))):
        java_list_wrapper.isEmpty()


def test_throws_if_not_unwrapped_and_passed_to_java_object(java_list_wrapper) -> None:
    with pytest.raises(
            TypeError,
            match="Trying to pass {} to a method that expects a JavaObject - did you forget to call unwrap()?".format(
                type(java_list_wrapper))):
        lst = KeanuContext()._gateway.jvm.java.util.ArrayList(java_list_wrapper)
