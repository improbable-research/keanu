from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.tensor import Tensor

import numpy as np
import numbers
from py4j.java_gateway import java_import

context = KeanuContext()
k = context.jvm_view()

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
            raise ValueError("Generic types in a tensor are not supported. Was given {}".format(type(scalar)))
