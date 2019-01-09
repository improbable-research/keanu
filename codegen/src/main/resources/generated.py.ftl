## This is a generated file. DO NOT EDIT.

from typing import Iterable
from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex, Double, Integer, Bool, vertex_constructor_param_types
from keanu.vartypes import (
    tensor_arg_types,
    shape_types
)

from .vertex_helpers import (
    do_double_vertex_cast,
)

context = KeanuContext()


def cast_to_double_vertex(input):
    do_double_vertex_cast(ConstantDouble, input)

def cast_to_integer_vertex(input):


def cast_to_boolean_vertex(input):


def cast_to_vertex(input):


def cast_to_double_tensor(input):


def cast_to_integer_tensor(input):


def cast_to_boolean_tensor(input):


def cast_to_double(input):


def cast_to_integer(input):


def cast_to_string(input):


def cast_to_long_array(input):


def cast_to_vertex_array(input):


<#list imports as import>
java_import(context.jvm_view(), "${import.packageName}")
</#list>
<#list constructors as constructor>


def ${constructor.pythonClass}(${constructor.pythonTypedParameters}) -> Vertex:
    ${constructor.docString}return ${constructor.pythonVertexClass}(context.jvm_view().${constructor.javaClass}, ${constructor.pythonParameters})
</#list>
