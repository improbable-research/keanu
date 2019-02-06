from py4j.java_gateway import java_import

from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.vertices.VertexLabel")


class _VertexLabel(JavaObjectWrapper):

    __separator = "."

    def __init__(self, first: str, *remainder: str):
        if len(remainder) == 0:
            java_object = k.jvm_view().VertexLabel(first)
        else:
            java_object = k.jvm_view().VertexLabel(first, k.to_java_string_array(remainder))
        super(_VertexLabel, self).__init__(java_object)

    @staticmethod
    def create_maybe_with_namespace(label: str) -> '_VertexLabel':
        """
        >>> l1 = _VertexLabel.create_maybe_with_namespace("foo")
        >>> l1.get_unqualified_name()
        'foo'
        >>> l1.get_qualified_name()
        'foo'
        >>> l2 = _VertexLabel.create_maybe_with_namespace("outer.inner.foo")
        >>> l2.get_unqualified_name()
        'foo'
        >>> l2.get_qualified_name()
        'outer.inner.foo'
        """
        if _VertexLabel.__separator in label:
            return _VertexLabel.create_with_namespace(label)
        else:
            return _VertexLabel(label)

    @staticmethod
    def create_with_namespace(label: str) -> '_VertexLabel':
        """
        >>> l = _VertexLabel.create_with_namespace("outer.inner.foo")
        >>> l.get_unqualified_name()
        'foo'
        >>> l.get_qualified_name()
        'outer.inner.foo'
        """
        if _VertexLabel.__separator not in label:
            raise ValueError('No namespace separator "{}" found in {}'.format(_VertexLabel.__separator, label))
        name_array = label.split(_VertexLabel.__separator)
        return _VertexLabel(name_array[0], *name_array[1:])

