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

        def get(self, index):
            return 100

    return JavaListWrapper([1, 2, 3])


def test_unwrapped_can_call_java_api(java_list_wrapper) -> None:
    assert not java_list_wrapper.unwrap().isEmpty()


def test_wrapped_can_call_python_api(java_list_wrapper) -> None:
    assert java_list_wrapper.get(0) == 100


def test_throws_if_there_is_no_java_api(java_list_wrapper) -> None:
    with pytest.raises(AttributeError, match="{} has no attribute missingMethod".format(type(java_list_wrapper))):
        java_list_wrapper.missingMethod()


def test_throws_if_not_unwrapped_and_passed_to_java_object(java_list_wrapper) -> None:
    with pytest.raises(
            TypeError,
            match="Trying to pass {} to a method that expects a JavaObject - did you forget to call unwrap()?".format(
                type(java_list_wrapper))):
        lst = KeanuContext()._gateway.jvm.java.util.ArrayList(java_list_wrapper)
