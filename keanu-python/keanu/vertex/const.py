from .generated import ConstantBool, ConstantInteger, ConstantDouble
from .base import Vertex
from typing import Callable, Any, Union
from keanu.tensor import cast_double, cast_integer, cast_bool, Tensor
from .vartypes import (
    numpy_types,
    tensor_arg_types,
    vertex_param_types,
    shape_types,
    primitive_types,
    runtime_int_types,
    runtime_float_types,
    runtime_bool_types,
    runtime_primitive_types,
    runtime_numpy_types,
    runtime_pandas_types,
    runtime_primitive_types
)
import numpy as np

def Const(t: tensor_arg_types) -> Vertex:
    if isinstance(t, runtime_numpy_types):
        ctor = __infer_const_ctor_from_ndarray(t)
        val = t
    elif isinstance(t, runtime_pandas_types):
        val = t.values
        ctor = __infer_const_ctor_from_ndarray(val)
    elif isinstance(t, runtime_primitive_types):
        ctor = __infer_const_ctor_from_scalar(t)
        val = np.array([[t]])
    else:
        raise NotImplementedError("Argument t must be either an ndarray or an instance of numbers.Number. Was given {} instead".format(type(t)))

    return ctor(Tensor(val))

def __infer_const_ctor_from_ndarray(ndarray: numpy_types) -> Callable:
    if len(ndarray) == 0:
        raise ValueError("Cannot infer type because the ndarray is empty")

    return __infer_const_ctor_from_scalar(ndarray.item(0))

def __infer_const_ctor_from_scalar(scalar: np.generic) -> Callable:
    if isinstance(scalar, runtime_bool_types):
        return ConstantBool
    elif isinstance(scalar, runtime_int_types):
        return ConstantInteger
    elif isinstance(scalar, runtime_float_types):
        return ConstantDouble
    else:
        raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(scalar)))

class Double(Vertex):
    def __init__(self, val: Any, *args: Union[vertex_param_types, shape_types]) -> None:
        if args:
            ctor = val
            val = ctor(*(Vertex.__parse_args(args)))
        else:
            val = ConstantDouble(cast_double(val).unwrap())

        super(Double, self).__init__(val)

    def observe(self, v: tensor_arg_types) -> None:
        self.unwrap().observe(cast_double(v))

    def set_value(self, v: tensor_arg_types) -> None:
        self.unwrap().setValue(cast_double(v))

    def set_and_cascade(self, v: tensor_arg_types) -> None:
        self.unwrap().setAndCascade(cast_double(v))


class Integer(Vertex):
    def __init__(self, val: Any, *args : Union[vertex_param_types, shape_types]) -> None:
        if args:
            ctor = val
            val = ctor(*(Vertex.__parse_args(args)))
        else:
            val = ConstantInteger(cast_integer(val).unwrap())

        super(Integer, self).__init__(val)

    def observe(self, v: tensor_arg_types) -> None:
        self.unwrap().observe(cast_integer(v))

    def set_value(self, v: tensor_arg_types) -> None:
        self.unwrap().setValue(cast_integer(v))

    def set_and_cascade(self, v: tensor_arg_types) -> None:
        self.unwrap().setAndCascade(cast_integer(v))


class Bool(Vertex):
    def __init__(self, val: Any, *args: Union[vertex_param_types, shape_types]) -> None:
        if args:
            ctor = val
            val = ctor(*(Vertex.__parse_args(args)))
        else:
            val = ConstantBool(cast_bool(val).unwrap())

        super(Bool, self).__init__(val)

    def observe(self, v: tensor_arg_types) -> None:
        self.unwrap().observe(cast_bool(v))

    def set_value(self, v: tensor_arg_types) -> None:
        self.unwrap().setValue(cast_bool(v))

    def set_and_cascade(self, v: tensor_arg_types) -> None:
        self.unwrap().setAndCascade(cast_bool(v))
