## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from keanu.vertex import Vertex

k = KeanuContext().jvm_view()


<#list imports as import>
java_import(k, "${import.packageName}")
</#list>
<#list constructors as constructor>


def ${constructor.pythonClass}(*args) -> k.${constructor.javaClass}:
    return Vertex(k.${constructor.javaClass}, args)
</#list>
