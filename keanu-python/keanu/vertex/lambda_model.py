from typing import Dict, Callable, Optional

from py4j.java_collections import JavaMap
from py4j.java_gateway import java_import

from keanu.context import KeanuContext
from keanu.functional import Consumer, Supplier
from keanu.vertex.base import Vertex
from keanu.vertex.label import _VertexLabel

context = KeanuContext()
java_import(context.jvm_view(), "io.improbable.keanu.vertices.model.LambdaModelVertex")


class LambdaModel(Vertex):

    def __init__(self,
                 inputs: Dict[str, Vertex],
                 executor: Callable,
                 update_values: Callable = None,
                 label: Optional[str] = None) -> None:
        self.vertices_wrapped = inputs
        vertex_map = LambdaModel.__to_java_map(inputs)
        self.executor = executor
        self.update_values = update_values or (lambda: self.vertices_wrapped)

        vertex = context.jvm_view().LambdaModelVertex(vertex_map, Consumer(self.__execute),
                                                      Supplier(lambda: self.__update_value()))
        super(LambdaModel, self).__init__(vertex, label)

    def __execute(self, vertices_unwrapped: JavaMap) -> None:
        self.vertices_wrapped = LambdaModel.__wrap(vertices_unwrapped)
        self.executor(self.vertices_wrapped)
        LambdaModel.__update_unwrapped_vertices(self.vertices_wrapped, vertices_unwrapped)

    def __update_value(self) -> JavaMap:
        values = self.update_values()
        return LambdaModel.__to_java_map(values)

    @staticmethod
    def __to_java_map(inputs: Dict[str, Vertex]) -> JavaMap:
        inputs_with_wrapped_keys = {_VertexLabel(k): v for k, v in inputs.items()}
        return context.to_java_map(inputs_with_wrapped_keys)

    @staticmethod
    def __wrap(vertices: JavaMap) -> Dict[str, Vertex]:
        return {k.getUnqualifiedName(): Vertex._from_java_vertex(v) for k, v in vertices.items()}

    @staticmethod
    def __update_unwrapped_vertices(vertices_wrapped: Dict[str, Vertex], vertices_unwrapped: JavaMap) -> None:
        for k, v in vertices_wrapped.items():
            vertices_unwrapped[_VertexLabel(k).unwrap()] = v.unwrap()

    def get_double_model_output_vertex(self, label: str) -> Vertex:
        label_unwrapped = _VertexLabel(label).unwrap()
        result = self.unwrap().getDoubleModelOutputVertex(label_unwrapped)
        return Vertex._from_java_vertex(result)
