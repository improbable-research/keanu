from py4j.java_gateway import java_import, JavaObject
from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from typing import List

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.vertices.VertexLabel")


class _VertexLabel(JavaObjectWrapper):

    def __init__(self, java_vertex_label: JavaObject) -> None:
        super(_VertexLabel, self).__init__(java_vertex_label)

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return other == self.get_unqualified_name()
        elif isinstance(other, _VertexLabel):
            return self.unwrap().equals(other.unwrap())
        else:
            return False

    def __hash__(self) -> int:
        return self.unwrap().hashCode()

    def __repr__(self) -> str:
        return self.get_qualified_name()

    def is_in_namespace(self, namespace: List[str]) -> bool:
        return self.unwrap().isInNamespace(k.to_java_string_array(namespace))

    def with_extra_namespace(self, top_level_namespace: str) -> '_VertexLabel':
        return _VertexLabel(self.unwrap().withExtraNamespace(top_level_namespace))

    def without_outer_namespace(self) -> '_VertexLabel':
        return _VertexLabel(self.unwrap().withoutOuterNamespace())

    def get_outer_namespace(self) -> str:
        outer_namespace = self.unwrap().getOuterNamespace()
        return outer_namespace.get() if outer_namespace.isPresent() else None

    def get_unqualified_name(self) -> str:
        return self.unwrap().getUnqualifiedName()

    def get_qualified_name(self) -> str:
        return self.unwrap().getQualifiedName()


class VertexLabel(_VertexLabel):

    def __init__(self, name: str, namespace=[]) -> None:
        java_object = k.jvm_view().VertexLabel(name, k.to_java_object_list(namespace))
        super(VertexLabel, self).__init__(java_object)
