from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vertex import Vertex
from keanu.generated import ConstantBool, ConstantDouble, ConstantInteger

import numpy as np
import numbers
from py4j.java_gateway import java_import

context = KeanuContext()
k = context.jvm_view()

def Const(t) -> Vertex:
    if isinstance(t, np.ndarray):
        ctor = __infer_const_from_ndarray(t)
        val = t
    elif isinstance(t, numbers.Number):
        ctor = __infer_const_from_scalar(t)
        val = np.array([[t]])
    else:
        raise NotImplementedError("Argument t must be either an ndarray or an instance of numbers.Number. Was given {} instead".format(type(t)))

    return ctor(Tensor(val))

def __infer_const_from_ndarray(ndarray):
    if len(ndarray) == 0:
        raise ValueError("Cannot infer type because the ndarray is empty")

    return __infer_const_from_scalar(ndarray.item(0))

def __infer_const_from_scalar(scalar):
    if isinstance(scalar, bool):
        return ConstantBool
    elif isinstance(scalar, int):
        return ConstantInteger
    elif isinstance(scalar, float):
        return ConstantDouble
    else:
        raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(scalar)))
