from .vartypes import (
    vertex_param_types,
    runtime_primitive_types,
    runtime_numpy_types,
    runtime_pandas_types
)

def cast_double(arg: vertex_param_types) -> vertex_param_types:
    if isinstance(arg, runtime_primitive_types):
        return float(arg)
    elif isinstance(arg, runtime_numpy_types):
        return arg.astype(float)
    elif isinstance(arg, runtime_pandas_types):
        return arg.values.astype(float)
    else:
        return arg

def cast_integer(arg: vertex_param_types) -> vertex_param_types:
    if isinstance(arg, runtime_primitive_types):
        return int(arg)
    elif isinstance(arg, runtime_numpy_types):
        return arg.astype(int)
    elif isinstance(arg, runtime_pandas_types):
        return arg.values.astype(int)
    else:
        return arg

def cast_bool(arg: vertex_param_types) -> vertex_param_types:
    if isinstance(arg, runtime_primitive_types):
        return bool(arg)
    elif isinstance(arg, runtime_numpy_types):
        return arg.astype(bool)
    elif isinstance(arg, runtime_pandas_types):
        return arg.values.astype(bool)
    else:
        return arg
