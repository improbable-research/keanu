## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex, Double, Integer, Bool, vertex_param_types
from keanu.vartypes import (
    tensor_arg_types,
    shape_types
)
from keanu.cast import cast_to_double, cast_to_integer, cast_to_bool

context = KeanuContext()


<#list imports as import>
java_import(context.jvm_view(), "${import.packageName}")
</#list>
<#list constructors as constructor>


def ${constructor.pythonClass}(${constructor.pythonTypedParameters}) -> Vertex:
    ${constructor.docString}return ${constructor.pythonVertexClass}(context.jvm_view().${constructor.javaClass}, ${constructor.pythonParameters})
</#list>
