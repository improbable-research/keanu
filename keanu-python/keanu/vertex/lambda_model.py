from py4j.java_gateway import java_import

from keanu.context import KeanuContext
from keanu.functional import Consumer, Supplier
from keanu.vertex.base import Vertex
from keanu.vertex.vertex_label import VertexLabel

context = KeanuContext()
java_import(context.jvm_view(), "io.improbable.keanu.vertices.model.LambdaModelVertex")


class LambdaModel(Vertex):

    def __init__(self, inputs, executor, update_value=None):
        self.executor = executor
        inputs_as_java_map = LambdaModel.__to_java_map(inputs)
        if update_value is None:
            update_value = lambda: inputs_as_java_map

        vertex = context.jvm_view().LambdaModelVertex(inputs_as_java_map, Consumer(self.__execute),
                                                      Supplier(update_value))
        super(LambdaModel, self).__init__(vertex)

    @staticmethod
    def __to_java_map(inputs):
        unwrapped_inputs = {VertexLabel(k).unwrap(): v for k, v in inputs.items()}
        return context.to_java_map(unwrapped_inputs)

    def __execute(self, vertices):
        vertices_wrapped = LambdaModel.__wrap(vertices)
        self.executor(vertices_wrapped)
        LambdaModel.__unwrap(vertices_wrapped, vertices)

    @staticmethod
    def __wrap(vertices):
        return {k.getUnqualifiedName(): Vertex(v) for k, v in vertices.items()}

    @staticmethod
    def __unwrap(vertices_wrapped, vertices_unwrapped):
        for k, v in vertices_wrapped.items():
            vertices_unwrapped[VertexLabel(k).unwrap()] = v.unwrap()

    def get_double_model_output_vertex(self, label):
        label = VertexLabel(label)
        result = self.unwrap().getDoubleModelOutputVertex(label.unwrap())
        return Vertex(result)
