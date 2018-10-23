from keanu.base import JavaCtor
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


class Const(JavaCtor):
    def __init__(self, t):
        if isinstance(t, np.ndarray):
            ctor = Const.__infer_const_from_ndarray(t)
            val = Tensor(t).unwrap()
        elif isinstance(t, numbers.Number):
            ctor = Const.__infer_const_from_scalar(t)
            val = t
        else:
            raise NotImplementedError("Argument t must be either an ndarray or an instance of numbers.Number. Was given {} instead".format(type(t)))

        super(Const, self).__init__(ctor, val)

    @staticmethod
    def __infer_const_from_ndarray(ndarray):
        if len(ndarray) == 0:
            raise ValueError("Cannot infer type because the ndarray is empty")

        return Const.__infer_const_from_scalar(ndarray.item(0))

    @staticmethod
    def __infer_const_from_scalar(scalar):
        if isinstance(scalar, bool):
            return k.ConstantBoolVertex
        elif isinstance(scalar, int):
            return k.ConstantIntegerVertex
        elif isinstance(scalar, float):
            return k.ConstantDoubleVertex
        else:
            raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(scalar)))
