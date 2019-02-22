import numpy as np
from numpy import ndarray
from py4j.java_gateway import java_import, JavaObject, JavaMember, is_instance_of
from typing import Any

from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.functional import Function
from .vartypes import (numpy_types, tensor_arg_types, primitive_types, runtime_int_types, runtime_float_types,
                       runtime_bool_types, runtime_numpy_types, runtime_pandas_types, runtime_primitive_types,
                       primitive_types)

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.tensor.dbl.DoubleTensor")
java_import(k.jvm_view(), "io.improbable.keanu.tensor.bool.BooleanTensor")
java_import(k.jvm_view(), "io.improbable.keanu.tensor.intgr.IntegerTensor")
java_import(k.jvm_view(), "io.improbable.keanu.util.Py4jByteArrayConverter")


class Tensor(JavaObjectWrapper):

    def __init__(self, t: tensor_arg_types) -> None:
        if isinstance(t, runtime_numpy_types):
            super(Tensor, self).__init__(Tensor.__get_tensor_from_ndarray(t))
        elif isinstance(t, runtime_pandas_types):
            super(Tensor, self).__init__(Tensor.__get_tensor_from_ndarray(t.values))
        elif isinstance(t, runtime_primitive_types):
            super(Tensor, self).__init__(Tensor.__get_tensor_from_scalar(t))
        else:
            raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(type(t)))

    def is_scalar(self) -> bool:
        return self.unwrap().isScalar()

    def scalar(self) -> primitive_types:
        return self.unwrap().scalar()

    def apply(self, lambda_function):
        return self.unwrap().apply(Function(lambda_function))

    def get_tensor_type(self) -> type:
        if "Double" in self._class:
            return float
        elif "Integer" in self._class:
            return int
        else:
            return bool

    @staticmethod
    def __get_tensor_from_ndarray(ndarray: numpy_types) -> JavaObject:

        ctor = Tensor.__infer_tensor_ctor_from_ndarray(ndarray)
        values = k.to_java_array(ndarray.flatten().tolist())
        shape = k.to_java_long_array(ndarray.shape)

        return ctor(values, shape)

    @staticmethod
    def __infer_tensor_ctor_from_ndarray(ndarray: numpy_types) -> JavaMember:
        if np.issubdtype(ndarray.dtype, np.bool_):
            return k.jvm_view().BooleanTensor.create
        elif np.issubdtype(ndarray.dtype, np.integer):
            return k.jvm_view().IntegerTensor.create
        elif np.issubdtype(ndarray.dtype, np.floating):
            return k.jvm_view().DoubleTensor.create
        else:
            raise NotImplementedError("Generic types in an ndarray are not supported. Was given {}".format(
                ndarray.dtype))

    @staticmethod
    def __get_tensor_from_scalar(scalar: primitive_types) -> JavaObject:
        if isinstance(scalar, runtime_bool_types):
            return k.jvm_view().BooleanTensor.scalar(bool(scalar))
        elif isinstance(scalar, runtime_int_types):
            return k.jvm_view().IntegerTensor.scalar(int(scalar))
        elif isinstance(scalar, runtime_float_types):
            return k.jvm_view().DoubleTensor.scalar(float(scalar))
        else:
            raise NotImplementedError("Generic types in a ndarray are not supported. Was given {}".format(type(scalar)))

    @staticmethod
    def _to_ndarray(java_tensor: JavaObject) -> Any:
        if (java_tensor.getRank() == 0):
            return np.array(java_tensor.scalar())
        else:
            return Tensor.__get_ndarray_from_tensor(java_tensor).reshape(java_tensor.getShape())

    @staticmethod
    def __get_ndarray_from_tensor(java_tensor) -> ndarray:
        # Performance is much better using byte arrays where possible.
        # https://stackoverflow.com/questions/39095994/fast-conversion-of-java-array-to-numpy-array-py4j
        if is_instance_of(k._gateway, java_tensor, "io.improbable.keanu.tensor.dbl.DoubleTensor"):
            byteArray = k.jvm_view().Py4jByteArrayConverter.toByteArray(java_tensor.asFlatDoubleArray())
            doubleArray = np.frombuffer(byteArray, np.float64)
            return doubleArray
        elif is_instance_of(k._gateway, java_tensor, "io.improbable.keanu.tensor.intgr.IntegerTensor"):
            byteArray = k.jvm_view().Py4jByteArrayConverter.toByteArray(java_tensor.asFlatIntegerArray())
            intArray = np.frombuffer(byteArray, np.int32)
            return intArray
        elif is_instance_of(k._gateway, java_tensor, "io.improbable.keanu.tensor.bool.BooleanTensor"):
            byteArray = k.jvm_view().Py4jByteArrayConverter.toByteArray(java_tensor.asFlatBooleanArray())
            boolArray = np.frombuffer(byteArray, bool)
            return boolArray
        else:
            return np.array(list(java_tensor.asFlatArray()))
