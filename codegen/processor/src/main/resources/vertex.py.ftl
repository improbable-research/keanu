## This is a generated file. DO NOT EDIT.

from py4j.java_gateway import java_import
from keanu.base import KeanuContext, Vertex

k = KeanuContext().jvm_view()
<#macro repeat count>
    <#list 1..count as n>
        <#nested n>
    </#list>
</#macro>
<@repeat count="${size}"?number ; n>


java_import(k, "${.vars["package" + n]}")

def ${.vars["py_class" + n]}(*args) -> k.${.vars["class" + n]}:
    return Vertex(k.${.vars["class" + n]}, args)
</@repeat>