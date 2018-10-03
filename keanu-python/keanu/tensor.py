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
java_import(k, "io.improbable.keanu.vertices.ConstantVertex")


class Const(JavaObjectWrapper):
    def __init__(self, t):
        if isinstance(t, np.ndarray):
            val = Tensor(t).unwrap()
        elif isinstance(t, numbers.Number):
            val = t
        else:
            raise ValueError("Argument t must be either a numpy array or an instance of numbers.Number")

        super(Const, self).__init__(k.ConstantVertex.of, val)


class Tensor(JavaObjectWrapper):
    def __init__(self, t):
        np_tensor = t if isinstance(t, np.ndarray) else np.array([t])

        values = context.to_java_array(np_tensor.flatten().tolist())
        shape = context.to_java_array(np_tensor.shape)

        super(Tensor, self).__init__(self.__infer_tensor_from_np_tensor(np_tensor), values, shape)

    @staticmethod
    def __infer_tensor_from_np_tensor(np_tensor):
        if len(np_tensor) == 0:
            raise ValueError("Cannot infer type because tensor is empty")

        if isinstance(np_tensor.item(0), int):
            return k.IntegerTensor.create
        elif isinstance(np_tensor.item(0), float):
            return k.DoubleTensor.create
        elif isinstance(np_tensor.item(0), bool):
            return k.BooleanTensor.create
        else:
            raise ValueError("Generic types in a tensor are not supported")

