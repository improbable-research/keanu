from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vertex.base import JavaObjectWrapper
from keanu.vertex import Const, Gaussian, LambdaModel, Vertex, VertexLabel
from keanu.functional import Function, Consumer


context = KeanuContext()

def plus_one(vertices): 
    input_vertex = vertices["in"]
    vertices[VertexLabel("out").unwrap()] = (Vertex(input_vertex) + 1.).unwrap()

def get_values(vertices):
    vertex_out = vertices[VertexLabel("out").unwrap()]
    value_map = {VertexLabel("out").unwrap() : JavaObjectWrapper(vertex_out.getValue()) }
    return context.to_java_map(value_map)

def test_you_can_create_a_lambda_model_vertex():
    v_in = Gaussian(1., 1.)
    inputs = context.to_java_map({ "in": v_in })

    executor = Consumer(plus_one)
    update_values = Function(get_values)
    model = LambdaModel(inputs, executor, update_values)
    v_out = model.get_double_model_output_vertex("out")

    v_in.set_value(1.)
    v_out.eval()
    assert v_out.get_value() == 2.

    v_in.set_value(1.1)
    v_out.eval()
    assert v_out.get_value() == 2.1
