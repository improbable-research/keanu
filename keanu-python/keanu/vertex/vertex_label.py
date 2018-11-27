from py4j.java_gateway import java_import
from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.vertices.VertexLabel")


class VertexLabel(JavaObjectWrapper):

    def __init__(self, name: str, namespace=[]):
        java_object = k.jvm_view().VertexLabel(name, k.to_java_object_list(namespace))
        super(VertexLabel, self).__init__(java_object)
