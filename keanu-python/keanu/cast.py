from .vartypes import (
    vertex_param_types,
    runtime_primitive_types,
    runtime_numpy_types,
    runtime_pandas_types
)

def __cast_to(arg: vertex_param_types, cast_to_type: type) -> vertex_param_types:
    if isinstance(arg, runtime_primitive_types):
        return cast_to_type(arg)
    elif isinstance(arg, runtime_numpy_types):
        return arg.astype(cast_to_type)
    elif isinstance(arg, runtime_pandas_types):
        return arg.values.astype(cast_to_type)
    else:
        return arg

def cast_to_double(arg: vertex_param_types) -> vertex_param_types:
    return __cast_to(arg, float)

def cast_to_integer(arg: vertex_param_types) -> vertex_param_types:
    return __cast_to(arg, int)

def cast_to_bool(arg: vertex_param_types) -> vertex_param_types:
    return __cast_to(arg, bool)
