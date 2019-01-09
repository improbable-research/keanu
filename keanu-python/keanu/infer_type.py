from typing import Callable

import numpy as np

from keanu.vartypes import (numpy_types, tensor_arg_types, runtime_numpy_types, runtime_pandas_types,
                            runtime_primitive_types, runtime_bool_types, runtime_int_types, runtime_float_types)


def infer_type_and_execute(value: tensor_arg_types, actions):
    return actions[__get_type_of_value(value)](value)


def __get_type_of_value(t):
    if isinstance(t, runtime_numpy_types):
        return __infer_type_from_ndarray(t)
    elif isinstance(t, runtime_pandas_types):
        return __infer_type_from_ndarray(t.values)
    elif isinstance(t, runtime_primitive_types):
        return __infer_type_from_scalar(t)
    else:
        raise NotImplementedError(
            "Argument t must be either an ndarray or an instance of numbers.Number. Was given {} instead".format(
                type(t)))


def __infer_type_from_ndarray(ndarray: numpy_types) -> Callable:
    if np.issubdtype(ndarray.dtype, np.bool_):
        return bool
    elif np.issubdtype(ndarray.dtype, np.integer):
        return int
    elif np.issubdtype(ndarray.dtype, np.floating):
        return float
    else:
        raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(ndarray.dtype))


def __infer_type_from_scalar(scalar: np.generic) -> Callable:
    if isinstance(scalar, runtime_bool_types):
        return bool
    elif isinstance(scalar, runtime_int_types):
        return int
    elif isinstance(scalar, runtime_float_types):
        return float
    else:
        raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(scalar)))
