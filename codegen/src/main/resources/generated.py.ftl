## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex
from keanu.vartypes import (
    vertex_arg_types,
    int_and_bool_vertex_arg_types,
    bool_vertex_arg_types,
    tensor_arg_types,
    int_and_bool_tensor_arg_types,
    bool_tensor_arg_types,
    shape_types
)

context = KeanuContext()


<#list imports as import>
java_import(context.jvm_view(), "${import.packageName}")
</#list>
<#list constructors as constructor>


def ${constructor.pythonClass}(${constructor.pythonTypedParameters}) -> Vertex:
    return Vertex(context.jvm_view().${constructor.javaClass}, ${constructor.pythonParameters})
</#list>
