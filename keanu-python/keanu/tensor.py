from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext

import numpy as np
from .vartypes import int_types, float_types, bool_types, primitive_types, pandas_types
from py4j.java_gateway import java_import

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.tensor.dbl.DoubleTensor")
java_import(k.jvm_view(), "io.improbable.keanu.tensor.bool.BooleanTensor")
java_import(k.jvm_view(), "io.improbable.keanu.tensor.intgr.IntegerTensor")

class Tensor(JavaObjectWrapper):
    def __init__(self, t):
        if isinstance(t, np.ndarray):
            super(Tensor, self).__init__(Tensor.__get_tensor_from_ndarray(t))
        elif isinstance(t, pandas_types):
            super(Tensor, self).__init__(Tensor.__get_tensor_from_ndarray(t.values))
        elif isinstance(t, primitive_types):
            super(Tensor, self).__init__(Tensor.__get_tensor_from_scalar(t))
        else:
            raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(t)))

    @staticmethod
    def __get_tensor_from_ndarray(ndarray):
        normalized_ndarray = Tensor.__ensure_rank_is_atleast_two(ndarray)

        ctor = Tensor.__infer_tensor_ctor_from_ndarray(normalized_ndarray)
        values = k.to_java_array(normalized_ndarray.flatten().tolist())
        shape = k.to_java_long_array(normalized_ndarray.shape)

        return ctor(values, shape)

    @staticmethod
    def __ensure_rank_is_atleast_two(ndarray):
        if len(ndarray.shape) == 1:
            return ndarray[..., None]
        else:
            return ndarray

    @staticmethod
    def __infer_tensor_ctor_from_ndarray(ndarray):
        if len(ndarray) == 0:
            raise ValueError("Cannot infer type because the ndarray is empty")

        if isinstance(ndarray.item(0), bool_types):
            return k.jvm_view().BooleanTensor.create
        elif isinstance(ndarray.item(0), int_types):
            return k.jvm_view().IntegerTensor.create
        elif isinstance(ndarray.item(0), float_types):
            return k.jvm_view().DoubleTensor.create
        else:
            raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(ndarray.item(0))))

    @staticmethod
    def __get_tensor_from_scalar(scalar):
        if isinstance(scalar, bool_types):
            return k.jvm_view().BooleanTensor.scalar(bool(scalar))
        elif isinstance(scalar, int_types):
            return k.jvm_view().IntegerTensor.scalar(int(scalar))
        elif isinstance(scalar, float_types):
            return k.jvm_view().DoubleTensor.scalar(float(scalar))
        else:
            raise NotImplementedError("Generic types in a ndarray are not supported. Was given {}".format(type(scalar)))

    @staticmethod
    def _to_ndarray(java_tensor):
        np_array = np.array(list(java_tensor.asFlatArray()))
        return np_array.reshape(java_tensor.getShape())
