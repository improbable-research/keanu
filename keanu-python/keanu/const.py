from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext

import numpy as np
import numbers
from py4j.java_gateway import java_import

context = KeanuContext()
k = context.jvm_view()

java_import(k, "io.improbable.keanu.tensor.dbl.DoubleTensor")
java_import(k, "io.improbable.keanu.tensor.bool.BooleanTensor")
java_import(k, "io.improbable.keanu.tensor.intgr.IntegerTensor")
java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex")
java_import(k, "io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex")
java_import(k, "io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex")


class Const(JavaObjectWrapper):
    def __init__(self, t):
        if isinstance(t, np.ndarray):
            ctor = Const.__infer_const_from_np_tensor(t)
            val = Tensor(t).unwrap()
        elif isinstance(t, numbers.Number):
            ctor = Const.__infer_const_from_scalar(t)
            val = t
        else:
            raise ValueError("Argument t must be either a numpy array or an instance of numbers.Number. Was given {} instead".format(type(t)))

        super(Const, self).__init__(ctor, val)

    @staticmethod
    def __infer_const_from_np_tensor(np_tensor):
        if len(np_tensor) == 0:
            raise ValueError("Cannot infer type because tensor is empty")

        return Const.__infer_const_from_scalar(np_tensor.item(0))

    @staticmethod
    def __infer_const_from_scalar(scalar):
        if isinstance(scalar, bool):
            return k.ConstantBoolVertex
        elif isinstance(scalar, int):
            return k.ConstantIntegerVertex
        elif isinstance(scalar, float):
            return k.ConstantDoubleVertex
        else:
            raise ValueError("Generic types in a tensor are not supported")


class Tensor(JavaObjectWrapper):
    def __init__(self, t):
        if isinstance(t, np.ndarray):
            normalized_tensor = Tensor.__ensure_rank_is_atleast_two(t)

            ctor = Tensor.__infer_tensor_from_np_tensor(normalized_tensor)
            values = context.to_java_array(normalized_tensor.flatten().tolist())
            shape = context.to_java_array(normalized_tensor.shape)

            super(Tensor, self).__init__(ctor, values, shape)
        elif isinstance(t, numbers.Number):
            super(Tensor, self).__init__(Tensor.__infer_tensor_from_scalar(t), t)
        else:
            raise ValueError("Generic types in a tensor are not supported")

    @staticmethod
    def __ensure_rank_is_atleast_two(np_tensor):
        if len(np_tensor.shape) == 1:
            return np_tensor[..., None]
        else:
            return np_tensor

    @staticmethod
    def __infer_tensor_from_np_tensor(np_tensor):
        if len(np_tensor) == 0:
            raise ValueError("Cannot infer type because tensor is empty")

        if isinstance(np_tensor.item(0), bool):
            return k.BooleanTensor.create
        elif isinstance(np_tensor.item(0), int):
            return k.IntegerTensor.create
        elif isinstance(np_tensor.item(0), float):
            return k.DoubleTensor.create
        else:
            raise ValueError("Generic types in a tensor are not supported")

    @staticmethod
    def __infer_tensor_from_scalar(scalar):
        if isinstance(scalar, bool):
            return k.BooleanTensor.scalar
        elif isinstance(scalar, int):
            return k.IntegerTensor.scalar
        elif isinstance(scalar, float):
            return k.DoubleTensor.scalar
        else:
            raise ValueError("Generic types in a tensor are not supported")

