from .base import (Vertex, cast_tensor_arg_to_double, cast_tensor_arg_to_integer, cast_tensor_arg_to_bool)
from keanu.tensor import Tensor
from keanu.context import KeanuContext

k = KeanuContext()

def do_vertex_cast(vertex_ctor, value):
    return value if isinstance(value, Vertex) else vertex_ctor(value)


def cast_to_vertex(input):
    return input


def cast_to_double_tensor(value):
    return Tensor(cast_tensor_arg_to_double(value))


def cast_to_integer_tensor(value):
    return Tensor(cast_tensor_arg_to_integer(value))


def cast_to_boolean_tensor(value):
    return Tensor(cast_tensor_arg_to_bool(value))


def cast_to_double(input):
    return float(input)


def cast_to_integer(input):
    return int(input)


def cast_to_string(input):
    return str(input)


def cast_to_long_array(input):
    return k.to_java_long_array(input)


def cast_to_vertex_array(input):
    return k.to_java_vertex_array(input)