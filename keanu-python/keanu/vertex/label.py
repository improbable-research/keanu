from py4j.java_gateway import java_import

from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.vertices.VertexLabel")


class _VertexLabel(JavaObjectWrapper):
    """
    This class is intended for private use - as a user you should only need to specify string values.
    It's used behind the scenes e.g. in :class:`keanu.vertex.lambda_model.LambdaModel`
    """

    __separator = "."

    def __init__(self, name: str):
        parts = name.split(_VertexLabel.__separator)

        if len(parts) == 1:
            java_object = k.jvm_view().VertexLabel(name)
        else:
            java_object = k.jvm_view().VertexLabel(parts[0], k.to_java_string_array(parts[1:]))
        super(_VertexLabel, self).__init__(java_object)

    def get_name(self) -> str:
        return self.unwrap().getQualifiedName()

    def __repr__(self) -> str:
        return self.get_name().split(".").__repr__()

    @staticmethod
    def create_from_list(*names: str) -> '_VertexLabel':
        if len(names) == 0:
            raise ValueError("You must pass in at least one string")

        names_joined = _VertexLabel.__separator.join(names)
        return _VertexLabel(names_joined)