## This is a generated file. DO NOT EDIT.

from typing import Iterable
from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex, Double, Integer, Boolean, vertex_constructor_param_types
from keanu.vartypes import (
    tensor_arg_types,
    shape_types
)
from .vertex_helpers import (
    do_vertex_cast
)

context = KeanuContext()


def cast_to_double_vertex(input):
    do_vertex_cast(ConstantDouble, input)


def cast_to_integer_vertex(input):
    do_vertex_cast(ConstantInteger, input)


def cast_to_boolean_vertex(input):
    pass


def cast_to_vertex(input):
    pass


def cast_to_double_tensor(input):
    pass


def cast_to_integer_tensor(input):
    pass


def cast_to_boolean_tensor(input):
    pass


def cast_to_double(input):
    pass


def cast_to_integer(input):
    pass


def cast_to_string(input):
    pass


def cast_to_long_array(input):
    pass


def cast_to_vertex_array(input):
    pass


<#list imports as import>
java_import(context.jvm_view(), "${import.packageName}")
</#list>
<#list constructors as constructor>


def ${constructor.pythonClass}(${constructor.pythonTypedParameters}) -> Vertex:
    ${constructor.docString}return ${constructor.pythonVertexClass}(context.jvm_view().${constructor.javaClass}, ${constructor.pythonParameters})
</#list>
