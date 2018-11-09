from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from typing import Any, TYPE_CHECKING
import numpy as np
from .vartypes import numpy_types, const_arg_types, primitive_types, runtime_int_types, runtime_float_types, runtime_bool_types, runtime_primitive_types, runtime_pandas_types, runtime_numpy_types, runtime_pandas_types, runtime_primitive_types
from py4j.java_gateway import java_import, JavaObject, JavaMember

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.tensor.dbl.DoubleTensor")
java_import(k.jvm_view(), "io.improbable.keanu.tensor.bool.BooleanTensor")
java_import(k.jvm_view(), "io.improbable.keanu.tensor.intgr.IntegerTensor")

class Tensor(JavaObjectWrapper):
    def __init__(self, t : const_arg_types) -> None:
        if isinstance(t, runtime_numpy_types):
            super(Tensor, self).__init__(Tensor.__get_tensor_from_ndarray(t))
        elif isinstance(t, runtime_pandas_types):
            super(Tensor, self).__init__(Tensor.__get_tensor_from_ndarray(t.values))
        elif isinstance(t, runtime_primitive_types):
            super(Tensor, self).__init__(Tensor.__get_tensor_from_scalar(t))
        else:
            raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(t)))

    @staticmethod
    def __get_tensor_from_ndarray(ndarray : numpy_types) -> Any:
        normalized_ndarray = Tensor.__ensure_rank_is_atleast_two(ndarray)

        ctor = Tensor.__infer_tensor_ctor_from_ndarray(normalized_ndarray)
        values = k.to_java_array(normalized_ndarray.flatten().tolist())
        shape = k.to_java_long_array(normalized_ndarray.shape)

        return ctor(values, shape)

    @staticmethod
    def __ensure_rank_is_atleast_two(ndarray : numpy_types) -> numpy_types:
        if len(ndarray.shape) == 1:
            return ndarray[..., None]
        else:
            return ndarray

    @staticmethod
    def __infer_tensor_ctor_from_ndarray(ndarray : numpy_types) -> Any:
        if len(ndarray) == 0:
            raise ValueError("Cannot infer type because the ndarray is empty")

        if isinstance(ndarray.item(0), runtime_bool_types):
            return k.jvm_view().BooleanTensor.create
        elif isinstance(ndarray.item(0), runtime_int_types):
            return k.jvm_view().IntegerTensor.create
        elif isinstance(ndarray.item(0), runtime_float_types):
            return k.jvm_view().DoubleTensor.create
        else:
            raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(ndarray.item(0))))

    @staticmethod
    def __get_tensor_from_scalar(scalar : primitive_types) -> Any:
        if isinstance(scalar, runtime_bool_types):
            return k.jvm_view().BooleanTensor.scalar(bool(scalar))
        elif isinstance(scalar, runtime_int_types):
            return k.jvm_view().IntegerTensor.scalar(int(scalar))
        elif isinstance(scalar, runtime_float_types):
            return k.jvm_view().DoubleTensor.scalar(float(scalar))
        else:
            raise NotImplementedError("Generic types in a ndarray are not supported. Was given {}".format(type(scalar)))

    @staticmethod
    def _to_ndarray(java_tensor : Any) -> numpy_types:
        np_array = np.array(list(java_tensor.asFlatArray()))
        return np_array.reshape(java_tensor.getShape())
