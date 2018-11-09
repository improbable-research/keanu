from keanu.context import KeanuContext
from keanu.vertex import Vertex
from keanu.vertex.vertex_label import VertexLabel
from keanu.tensor import Tensor
from keanu.functional import Consumer, Function
from keanu.vertex.base import JavaObjectWrapper
from py4j.java_gateway import java_import


context = KeanuContext()
java_import(context.jvm_view(), "io.improbable.keanu.vertices.model.LambdaModelVertex")

def _extract_values(vertices):
    value_map = {k: JavaObjectWrapper(v.getValue()) for k, v in vertices.items()}
    return context.to_java_map(value_map)

class LambdaModel(Vertex):
    def __init__(self, 
    inputs: map, 
    executor, 
    update_value = _extract_values
    ) -> Vertex:
        self.inputs = context.to_java_map(inputs)
        self.executor = executor
        self.update_value = Function(update_value)
        val = context.jvm_view().LambdaModelVertex(self.inputs, Consumer(self.__execute), self.update_value)
        super(LambdaModel, self).__init__(val)

    def __execute(self, vertices):
        vertices_wrapped = self.__wrap(vertices)
        self.executor(vertices_wrapped)
        self.__unwrap(vertices_wrapped, vertices)

    def __wrap(self, vertices):
        return {k: Vertex(v) for k, v in vertices.items()}

    def __unwrap(self, vertices_wrapped, vertices_unwrapped):
        for k, v in vertices_wrapped.items():
            vertices_unwrapped[k] = v.unwrap()

    def get_double_model_output_vertex(self, label : str):
        label = VertexLabel(label)
        result = self.unwrap().getDoubleModelOutputVertex(label.unwrap())
        return Vertex(result)


