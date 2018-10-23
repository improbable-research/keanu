from keanu.base import JavaCtor
from keanu.context import KeanuContext

import numpy as np
import numbers
from py4j.java_gateway import java_import

context = KeanuContext()
k = context.jvm_view()

java_import(k, "io.improbable.keanu.tensor.dbl.DoubleTensor")
java_import(k, "io.improbable.keanu.tensor.bool.BooleanTensor")
java_import(k, "io.improbable.keanu.tensor.intgr.IntegerTensor")

class Tensor(JavaCtor):
    def __init__(self, t):
        if isinstance(t, np.ndarray):
            normalized_ndarray = Tensor.__ensure_rank_is_atleast_two(t)

            ctor = Tensor.__infer_tensor_from_ndarray(normalized_ndarray)
            values = context.to_java_array(normalized_ndarray.flatten().tolist())
            shape = context.to_java_long_array(normalized_ndarray.shape)

            super(Tensor, self).__init__(ctor, values, shape)
        elif isinstance(t, numbers.Number):
            super(Tensor, self).__init__(Tensor.__infer_tensor_from_scalar(t), t)
        else:
            raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(t)))

    @staticmethod
    def __ensure_rank_is_atleast_two(ndarray):
        if len(ndarray.shape) == 1:
            return ndarray[..., None]
        else:
            return ndarray

    @staticmethod
    def __infer_tensor_from_ndarray(ndarray):
        if len(ndarray) == 0:
            raise ValueError("Cannot infer type because the ndarray is empty")

        if isinstance(ndarray.item(0), bool):
            return k.BooleanTensor.create
        elif isinstance(ndarray.item(0), int):
            return k.IntegerTensor.create
        elif isinstance(ndarray.item(0), float):
            return k.DoubleTensor.create
        else:
            raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(ndarray.item(0))))

    @staticmethod
    def __infer_tensor_from_scalar(scalar):
        if isinstance(scalar, bool):
            return k.BooleanTensor.scalar
        elif isinstance(scalar, int):
            return k.IntegerTensor.scalar
        elif isinstance(scalar, float):
            return k.DoubleTensor.scalar
        else:
            raise NotImplementedError("Generic types in a ndarray are not supported. Was given {}".format(type(scalar)))

