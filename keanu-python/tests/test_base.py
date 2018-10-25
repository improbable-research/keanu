import keanu as kn
import pytest
from py4j.java_gateway import java_import

@pytest.fixture
def java_list_wrapper():
    class JavaListWrapper(kn.JavaObjectWrapper):
        def __init__(self, values):
            super(JavaListWrapper, self).__init__(kn.KeanuContext().to_java_list(values))

        def get(self, index):
            return 4

        def index_of(self, value):
            return 100

    return JavaListWrapper([1, 2, 3])

def test_java_object_wrapper_can_call_java_api_with_no_python_impl(java_list_wrapper):
    assert not java_list_wrapper.isEmpty()

def test_java_object_wrapper_convert_python_name_to_java_name(java_list_wrapper):
    assert not java_list_wrapper.is_empty()

def test_java_object_wrapper_can_call_python_api(java_list_wrapper):
    assert java_list_wrapper.index_of(1) == 100

def test_java_object_wrapper_convert_java_name_to_python_name(java_list_wrapper):
    assert java_list_wrapper.indexOf(1) == 100

def test_java_object_wrapper_attr_override(java_list_wrapper):
    assert java_list_wrapper.get(0) == 4
