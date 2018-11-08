## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from .base import Vertex

context = KeanuContext()


<#list imports as import>
java_import(context.jvm_view(), "${import.packageName}")
</#list>
<#list constructors as constructor>


def ${constructor.pythonClass}(${constructor.pythonParameters}) -> context.jvm_view().${constructor.javaClass}:
    ${constructor.docString}return Vertex(context.jvm_view().${constructor.javaClass}, ${constructor.pythonParameters})
</#list>
