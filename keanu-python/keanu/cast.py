from .vertex.base import vertex_constructor_param_types
from .vartypes import (tensor_arg_types, runtime_tensor_arg_types, runtime_primitive_types, runtime_numpy_types,
                       runtime_pandas_types)


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


def cast_to_double(arg: vertex_constructor_param_types) -> vertex_constructor_param_types:
    return cast_tensor_arg_to_double(arg) if isinstance(arg, runtime_tensor_arg_types) else arg


def cast_to_integer(arg: vertex_constructor_param_types) -> vertex_constructor_param_types:
    return cast_tensor_arg_to_integer(arg) if isinstance(arg, runtime_tensor_arg_types) else arg


def cast_to_bool(arg: vertex_constructor_param_types) -> vertex_constructor_param_types:
    return cast_tensor_arg_to_bool(arg) if isinstance(arg, runtime_tensor_arg_types) else arg
