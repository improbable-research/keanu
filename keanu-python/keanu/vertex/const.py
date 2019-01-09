from typing import Callable

import numpy as np

from keanu.tensor import Tensor
from keanu.vartypes import (numpy_types, tensor_arg_types, runtime_numpy_types, runtime_pandas_types,
                            runtime_primitive_types, runtime_bool_types, runtime_int_types, runtime_float_types)
from .base import Vertex


def create_constant_boolean(value):


def create_constant_double(value):


def create_constant_integer(value):



def Const(t: tensor_arg_types) -> Vertex:
    if isinstance(t, runtime_numpy_types):
        ctor = __infer_const_ctor_from_ndarray(t)
        val = t
    elif isinstance(t, runtime_pandas_types):
        val = t.values
        ctor = __infer_const_ctor_from_ndarray(val)
    elif isinstance(t, runtime_primitive_types):
        ctor = __infer_const_ctor_from_scalar(t)
        val = t
    else:
        raise NotImplementedError(
            "Argument t must be either an ndarray or an instance of numbers.Number. Was given {} instead".format(
                type(t)))

    return ctor(Tensor(val))


def __infer_const_ctor_from_ndarray(ndarray: numpy_types) -> Callable:
    if np.issubdtype(ndarray.dtype, np.bool_):
        return create_constant_boolean
    elif np.issubdtype(ndarray.dtype, np.integer):
        return create_constant_integer
    elif np.issubdtype(ndarray.dtype, np.floating):
        return create_constant_double
    else:
        raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(ndarray.dtype))


def __infer_const_ctor_from_scalar(scalar: np.generic) -> Callable:
    if isinstance(scalar, runtime_bool_types):
        return create_constant_boolean
    elif isinstance(scalar, runtime_int_types):
        return create_constant_integer
    elif isinstance(scalar, runtime_float_types):
        return create_constant_double
    else:
        raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(scalar)))
