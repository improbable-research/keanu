import numpy as np
from py4j.java_gateway import java_import

from keanu.base import JavaObjectWrapper
from keanu.vartypes import (
    tensor_arg_types,
    runtime_primitive_types,
    runtime_numpy_types,
    runtime_pandas_types
)
from typing import Callable

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.tensor.dbl.DoubleTensor")
java_import(k.jvm_view(), "io.improbable.keanu.tensor.bool.BooleanTensor")
java_import(k.jvm_view(), "io.improbable.keanu.tensor.intgr.IntegerTensor")

def cast_double(arg: tensor_arg_types) -> __Tensor:
    if isinstance(arg, runtime_primitive_types):
        return __Tensor(k.jvm_view().DoubleTensor.scalar(float(arg))
    elif isinstance(arg, runtime_numpy_types):
        return __Tensor(__get_java_tensor(k.jvm_view().DoubleTensor.create, arg.astype(float))
    elif isinstance(arg, runtime_pandas_types):
        return __Tensor(__get_java_tensor(k.jvm_View().DoubleTensor.create, arg.values.astype(float))
    else:
        raise NotImplementedError

def cast_integer(arg: tensor_arg_types) -> __Tensor:
    if isinstance(arg, runtime_primitive_types):
        return __Tensor(k.jvm_view().IntegerTensor.scalar(int(arg))
    elif isinstance(arg, runtime_numpy_types):
        return __Tensor(__get_java_tensor(k.jvm_view().IntegerTensor.create, arg.astype(int))
    elif isinstance(arg, runtime_pandas_types):
        return __Tensor(__get_java_tensor(k.jvm_View().IntegerTensor.create, arg.values.astype(int))
    else:
        raise NotImplementedError

def cast_bool(arg: tensor_arg_types) -> __Tensor:
    if isinstance(arg, runtime_primitive_types):
        return __Tensor(k.jvm_view().BooleanTensor.scalar(bool(arg))
    elif isinstance(arg, runtime_numpy_types):
        return __Tensor(__get_java_tensor(k.jvm_view().BooleanTensor.create, arg.astype(bool))
    elif isinstance(arg, runtime_pandas_types):
        return __Tensor(__get_java_tensor(k.jvm_View().BooleanTensor.create, arg.values.astype(bool))
    else:
        raise NotImplementedError

def _to_ndarray(java_tensor: Any) -> numpy_types:
    np_array = np.array(list(java_tensor.asFlatArray()))
    return np_array.reshape(java_tensor.getShape())

def __ensure_rank_is_atleast_two(ndarray: numpy_types) -> numpy_types:
    if len(ndarray.shape) == 1:
        return ndarray[..., None]
    else:
        return ndarray

def __get_java_tensor(ctor: Callable, ndarray: numpy_types) -> Any:
    normalized_ndarray = Tensor.__ensure_rank_is_atleast_two(ndarray)

    values = k.to_java_array(normalized_ndarray.flatten().tolist())
    shape = k.to_java_long_array(normalized_ndarray.shape)

    return ctor(values, shape)

class __Tensor(JavaObjectWrapper):
    def __init__(self, java_tensor):
        super(__Tensor, self).__init__(java_tensor)
