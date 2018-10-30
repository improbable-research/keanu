## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from keanu.vertex import Vertex

k = KeanuContext()


<#list imports as import>
java_import(k.jvm_view(), "${import.packageName}")
</#list>
<#list constructors as constructor>


def ${constructor.pythonClass}(*args) -> k.jvm_view().${constructor.javaClass}:
    return Vertex(k.jvm_view().${constructor.javaClass}, args)
</#list>
