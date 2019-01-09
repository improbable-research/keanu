from .base import Vertex
from keanu.tensor import Tensor
from keanu.vartypes import (tensor_arg_types, runtime_tensor_arg_types, runtime_primitive_types, runtime_numpy_types,
                       runtime_pandas_types)

def do_vertex_cast(vertex_ctor, value):
    return value if isinstance(value, Vertex) else vertex_ctor(value)


def cast_to_vertex(input):
    return input


def __cast_to(arg: tensor_arg_types, cast_to_type: type) -> tensor_arg_types:
    if isinstance(arg, runtime_primitive_types):
        return cast_to_type(arg)
    elif isinstance(arg, runtime_numpy_types):
        return arg.astype(cast_to_type)
    elif isinstance(arg, runtime_pandas_types):
        return arg.values.astype(cast_to_type)
    else:
        raise TypeError("Cannot cast {} to {}".format(type(arg), cast_to_type))


def cast_tensor_arg_to_double(arg: tensor_arg_types) -> tensor_arg_types:
    return __cast_to(arg, float)


def cast_tensor_arg_to_integer(arg: tensor_arg_types) -> tensor_arg_types:
    return __cast_to(arg, int)


def cast_tensor_arg_to_bool(arg: tensor_arg_types) -> tensor_arg_types:
    return __cast_to(arg, bool)


def cast_to_double_tensor(value):
    return Tensor(cast_tensor_arg_to_double(value))


def cast_to_integer_tensor(value):
    return Tensor(cast_tensor_arg_to_integer(value))


def cast_to_boolean_tensor(value):
    return Tensor(cast_tensor_arg_to_bool(value))


def cast_to_double(input):
    pass


def cast_to_integer(input):
    pass


def cast_to_string(input):
    pass


def cast_to_long_array(input):
    pass


def cast_to_vertex_array(input):
    pass