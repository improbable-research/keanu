## This is a generated file. DO NOT EDIT.

from typing import Collection, Optional
from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex, Double, Integer, Boolean, vertex_constructor_param_types
from keanu.vartypes import (
    tensor_arg_types,
    shape_types
)
from .vertex_casting import (
    do_vertex_cast,
    do_inferred_vertex_cast,
    cast_to_double_tensor,
    cast_to_integer_tensor,
    cast_to_boolean_tensor,
    cast_to_double,
    cast_to_integer,
    cast_to_string,
    cast_to_boolean,
    cast_to_long_array,
    cast_to_int_array,
    cast_to_vertex_array,
)

context = KeanuContext()


def cast_to_double_vertex(input: vertex_constructor_param_types) -> Vertex:
    return do_vertex_cast(ConstantDouble, input)


def cast_to_integer_vertex(input: vertex_constructor_param_types) -> Vertex:
    return do_vertex_cast(ConstantInteger, input)


def cast_to_boolean_vertex(input: vertex_constructor_param_types) -> Vertex:
    return do_vertex_cast(ConstantBoolean, input)


def cast_to_vertex(input: vertex_constructor_param_types) -> Vertex:
    return do_inferred_vertex_cast({bool: ConstantBoolean, int: ConstantInteger, float: ConstantDouble}, input)


<#list imports as import>
java_import(context.jvm_view(), "${import.packageName}")
</#list>
<#list constructors as javaConstructor>


def ${javaConstructor.pythonClass}(${javaConstructor.pythonTypedParameters}) -> Vertex:
    ${javaConstructor.docString}return ${javaConstructor.pythonVertexClass}(context.jvm_view().${javaConstructor.javaClass}, ${javaConstructor.pythonParameters})
</#list>
