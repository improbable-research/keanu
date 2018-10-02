from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext

import numpy as np
from py4j.java_gateway import java_import

context = KeanuContext()
k = context.jvm_view()

java_import(k, "io.improbable.keanu.tensor.dbl.DoubleTensor")
java_import(k, "io.improbable.keanu.tensor.bool.BooleanTensor")
java_import(k, "io.improbable.keanu.tensor.intgr.IntegerTensor")
java_import(k, "io.improbable.keanu.tensor.generic.GenericTensor")

java_import(k, "io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex")
java_import(k, "io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex")
java_import(k, "io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex")
java_import(k, "io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantGenericVertex")


class Const(JavaObjectWrapper):
    def __init__(self, t):
        np_tensor = t if isinstance(t, np.ndarray) else np.array([t])

        ctor = self._infer_vertex_from_np_tensor(np_tensor)
        tensor = Tensor(np_tensor)

        super(Const, self).__init__(ctor, tensor.unwrap())

    @staticmethod
    def _infer_vertex_from_np_tensor(np_tensor):
        if len(np_tensor) == 0:
            raise ValueError("Cannot infer type because tensor is empty")

        if isinstance(np_tensor.item(0), int):
            return k.ConstantIntegerVertex
        elif isinstance(np_tensor.item(0), float):
            return k.ConstantDoubleVertex
        elif isinstance(np_tensor.item(0), bool):
            return k.ConstantBoolVertex
        else:
            return k.ConstantGenericVertex


class Tensor(JavaObjectWrapper):
    def __init__(self, t):
        np_tensor = t if isinstance(t, np.ndarray) else np.array([t])

        values = context.to_java_array(np_tensor.flatten().tolist())
        shape = context.to_java_array(np_tensor.shape)

        super(Tensor, self).__init__(self._infer_tensor_from_np_tensor(np_tensor), values, shape)

    @staticmethod
    def _infer_tensor_from_np_tensor(np_tensor):
        if len(np_tensor) == 0:
            raise ValueError("Cannot infer type because tensor is empty")

        if isinstance(np_tensor.item(0), int):
            return k.IntegerTensor.create
        elif isinstance(np_tensor.item(0), float):
            return k.DoubleTensor.create
        elif isinstance(np_tensor.item(0), bool):
            return k.BooleanTensor.create
        else:
            return k.GenericTensor

