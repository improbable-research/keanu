from typing import Callable, Dict, Any, Collection
from .base import (Vertex, cast_tensor_arg_to_double, cast_tensor_arg_to_integer, cast_tensor_arg_to_bool,
                   vertex_constructor_param_types)
from keanu.tensor import Tensor
from keanu.context import KeanuContext
from keanu.infer_type import infer_type_and_execute
from keanu.vartypes import tensor_arg_types
from py4j.java_collections import JavaArray

k = KeanuContext()


def do_vertex_cast(vertex_ctor: Callable, value: vertex_constructor_param_types) -> Vertex:
    return value if isinstance(value, Vertex) else vertex_ctor(value)


def do_generic_vertex_cast(ctors: Dict[type, Callable], value: vertex_constructor_param_types) -> Vertex:
    return value if isinstance(value, Vertex) else infer_type_and_execute(value, ctors)


def cast_to_double_tensor(value: tensor_arg_types) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(cast_tensor_arg_to_double(value))


def cast_to_integer_tensor(value: tensor_arg_types) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(cast_tensor_arg_to_integer(value))


def cast_to_boolean_tensor(value: tensor_arg_types) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(cast_tensor_arg_to_bool(value))


def cast_to_double(input: tensor_arg_types) -> float:
    return float(input)


def cast_to_integer(input: tensor_arg_types) -> int:
    return int(input)


def cast_to_string(input: Any) -> str:
    return str(input)


def cast_to_long_array(input: Collection[int]) -> JavaArray:
    return k.to_java_long_array(input)


def cast_to_int_array(input: Collection[int]) -> JavaArray:
    return k.to_java_int_array(input)


def cast_to_vertex_array(input: Collection[Vertex]) -> JavaArray:
    return k.to_java_vertex_array(input)
