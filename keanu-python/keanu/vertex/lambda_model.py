from py4j.java_gateway import java_import

from keanu.context import KeanuContext
from keanu.functional import Consumer, Supplier
from keanu.vertex.base import Vertex
from keanu.vertex.vertex_label import VertexLabel

context = KeanuContext()
java_import(context.jvm_view(), "io.improbable.keanu.vertices.model.LambdaModelVertex")


class LambdaModel(Vertex):

    def __init__(self, inputs, executor, update_values=None):
        self.vertices_wrapped = inputs
        vertex_map = LambdaModel.__to_java_map(inputs)
        self.executor = executor
        self.update_values = update_values or (lambda: self.vertices_wrapped)

        vertex = context.jvm_view().LambdaModelVertex(vertex_map, Consumer(self.__execute),
                                                      Supplier(lambda: self.__update_value()))
        super(LambdaModel, self).__init__(vertex)

    def __execute(self, vertices_unwrapped):
        self.vertices_wrapped = LambdaModel.__wrap(vertices_unwrapped)
        self.executor(self.vertices_wrapped)
        LambdaModel.__update_unwrapped_vertices(self.vertices_wrapped, vertices_unwrapped)

    def __update_value(self):
        values = self.update_values()
        return LambdaModel.__to_java_map(values)

    @staticmethod
    def __to_java_map(inputs):
        unwrapped_inputs = {VertexLabel(k).unwrap(): v for k, v in inputs.items()}
        return context.to_java_map(unwrapped_inputs)

    @staticmethod
    def __wrap(vertices):
        return {k.getUnqualifiedName(): Vertex(v) for k, v in vertices.items()}

    @staticmethod
    def __update_unwrapped_vertices(vertices_wrapped, vertices_unwrapped):
        for k, v in vertices_wrapped.items():
            vertices_unwrapped[VertexLabel(k).unwrap()] = v.unwrap()

    def get_double_model_output_vertex(self, label):
        label = VertexLabel(label)
        result = self.unwrap().getDoubleModelOutputVertex(label.unwrap())
        return Vertex(result)
