## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex
from keanu.vartypes import mypy_vertex_arg_types, mypy_shape_types

context = KeanuContext()


<#list imports as import>
java_import(context.jvm_view(), "${import.packageName}")
</#list>
<#list constructors as constructor>


def ${constructor.pythonClass}(${constructor.pythonTypedParameters}) -> Vertex:
    return Vertex(context.jvm_view().${constructor.javaClass}, ${constructor.pythonParameters})
</#list>
