import collections
from typing import List, Tuple, Iterator, Union, SupportsRound, Optional, Any, Dict
from typing import cast as typing_cast

import numpy as np
from py4j.java_collections import JavaList, JavaArray
from py4j.java_gateway import JavaObject, JavaMember

import keanu as kn
from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vertex.label import _VertexLabel
from keanu.vartypes import (tensor_arg_types, wrapped_java_types, shape_types, numpy_types, runtime_wrapped_java_types,
                            runtime_primitive_types, runtime_numpy_types, runtime_pandas_types, vertex_label_types)

k = KeanuContext()

vertex_operation_param_types = Union['Vertex', tensor_arg_types]
vertex_constructor_param_types = Union['Vertex', tensor_arg_types, wrapped_java_types, str]


class Vertex(JavaObjectWrapper, SupportsRound['Vertex']):

    def __init__(self, val_or_ctor: Union[JavaMember, JavaObject],
                 *args: Union[vertex_constructor_param_types, shape_types], **kwargs: Dict[str, Any]) -> None:
        val: JavaObject
        if args:
            ctor = val_or_ctor
            val = ctor(*(Vertex.__parse_args(args)))
        else:
            val = typing_cast(JavaObject, val_or_ctor)

        Vertex.__handle_optional_params(val, **kwargs)
        super(Vertex, self).__init__(val)

    def cast(self, v: tensor_arg_types) -> tensor_arg_types:
        return v

    def __hash__(self) -> int:
        return hash(self.get_id())

    def observe(self, v: tensor_arg_types) -> None:
        self.unwrap().observe(Tensor(self.cast(v)).unwrap())

    def set_value(self, v: tensor_arg_types) -> None:
        self.unwrap().setValue(Tensor(self.cast(v)).unwrap())

    def set_and_cascade(self, v: tensor_arg_types) -> None:
        self.unwrap().setAndCascade(Tensor(self.cast(v)).unwrap())

    def set_label(self, label: vertex_label_types) -> None:
        Vertex.__set_label_to_java_vertex(self.unwrap(), label)

    def sample(self) -> numpy_types:
        return Tensor._to_ndarray(self.unwrap().sample())

    def get_value(self) -> numpy_types:
        return Tensor._to_ndarray(self.unwrap().getValue())

    def get_connected_graph(self) -> Iterator['Vertex']:
        return Vertex._to_generator(self.unwrap().getConnectedGraph())

    def get_id(self) -> Tuple[JavaObject, ...]:
        return Vertex._get_python_id(self.unwrap())

    def get_label(self) -> _VertexLabel:
        return Vertex._get_python_label(self.unwrap())

    """
    __array_ufunc__ is a NumPy thing that enables you to intercept and handle the numpy operation.
    Without this the right operators would fail.
    See https://docs.scipy.org/doc/numpy-1.13.0/neps/ufunc-overrides.html
    """

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, input0: numpy_types, input1: 'Vertex') -> 'Vertex':
        methods = {
            "equal": Vertex.__eq__,
            "not_equal": Vertex.__ne__,
            "add": Vertex.__radd__,
            "subtract": Vertex.__rsub__,
            "multiply": Vertex.__rmul__,
            "power": Vertex.__rpow__,
            "true_divide": Vertex.__rtruediv__,
            "floor_divide": Vertex.__rfloordiv__,
            "greater": Vertex.__lt__,
            "greater_equal": Vertex.__le__,
            "less": Vertex.__gt__,
            "less_equal": Vertex.__ge__,
        }
        if method == "__call__":
            try:
                dispatch_method = methods[ufunc.__name__]
                result = dispatch_method(input1, input0)
                return result
            except KeyError:
                raise NotImplementedError("NumPy ufunc of type %s not implemented" % ufunc.__name__)
        else:
            raise NotImplementedError("NumPy ufunc method %s not implemented" % method)

    def __add__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Addition(self, other)

    def __radd__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Addition(other, self)

    def __sub__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Difference(self, other)

    def __rsub__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Difference(other, self)

    def __mul__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Multiplication(self, other)

    def __rmul__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Multiplication(other, self)

    def __pow__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Power(self, other)

    def __rpow__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Power(other, self)

    def __truediv__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Division(self, other)

    def __rtruediv__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Division(other, self)

    def __floordiv__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.IntegerDivision(self, other)

    def __rfloordiv__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.IntegerDivision(other, self)

    def __eq__(  # type: ignore # see https://github.com/python/mypy/issues/2783
            self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Equals(self, other)

    def __ne__(  # type: ignore # see https://github.com/python/mypy/issues/2783
            self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.NotEquals(self, other)

    def __gt__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.GreaterThan(self, other)

    def __ge__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.GreaterThanOrEqual(self, other)

    def __lt__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.LessThan(self, other)

    def __le__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.LessThanOrEqual(self, other)

    def __abs__(self) -> 'Vertex':
        return kn.vertex.generated.Abs(self)

    def __round__(self, ndigits: Optional[int] = 0) -> 'Vertex':
        if ndigits != 0:
            raise NotImplementedError("Keanu only supports rounding to 0 digits")
        return kn.vertex.generated.Round(self)

    def __floor__(self) -> 'Vertex':
        return kn.vertex.generated.Floor(self)

    def __ceil__(self) -> 'Vertex':
        return kn.vertex.generated.Ceil(self)

    @staticmethod
    def __parse_args(args: Tuple[Union[vertex_constructor_param_types, shape_types], ...]) -> List[JavaObject]:
        return list(map(Vertex.__parse_arg, args))

    @staticmethod
    def __parse_arg(arg: Union[vertex_constructor_param_types, shape_types]) -> JavaObject:
        if isinstance(arg, runtime_wrapped_java_types):
            return arg.unwrap()
        elif isinstance(arg, collections.Collection) and all(isinstance(x, runtime_primitive_types) for x in arg):
            return k.to_java_long_array(arg)
        elif isinstance(arg, runtime_primitive_types) or isinstance(arg, JavaObject):
            return arg
        else:
            raise ValueError("Can't parse generic argument. Was given {}".format(type(arg)))

    @staticmethod
    def __handle_optional_params(val: JavaObject, **kwargs: Dict[str, Any]) -> None:
        if "label" in kwargs:
            casted_kwarg = typing_cast(vertex_label_types, kwargs["label"])
            Vertex.__set_label_to_java_vertex(val, casted_kwarg)

    @staticmethod
    def __set_label_to_java_vertex(java_vertex, label: vertex_label_types) -> None:
        if isinstance(label, _VertexLabel):
            java_vertex.setLabel(label.unwrap())
        elif isinstance(label, str):
            java_vertex.setLabel(label)
        else:
            raise TypeError("label must be str or VertexLabel.")

    @staticmethod
    def _to_generator(java_vertices: Union[JavaList, JavaArray]) -> Iterator['Vertex']:
        return (Vertex(java_vertex) for java_vertex in java_vertices)

    @staticmethod
    def _get_python_id(java_vertex: JavaObject) -> Tuple[JavaObject, ...]:
        return tuple(java_vertex.getId().getValue())

    @staticmethod
    def _get_python_label(java_vertex: JavaObject) -> _VertexLabel:
        return _VertexLabel(java_vertex.getLabel())


class Double(Vertex):

    def cast(self, v: tensor_arg_types) -> tensor_arg_types:
        return cast_tensor_arg_to_double(v)


class Integer(Vertex):

    def cast(self, v: tensor_arg_types) -> tensor_arg_types:
        return cast_tensor_arg_to_integer(v)


class Boolean(Vertex):

    def cast(self, v: tensor_arg_types) -> tensor_arg_types:
        return cast_tensor_arg_to_boolean(v)


def __cast_to(arg: tensor_arg_types, cast_to_type: type) -> tensor_arg_types:
    if isinstance(arg, runtime_primitive_types):
        return cast_to_type(arg)
    elif isinstance(arg, runtime_numpy_types):
        return arg.astype(cast_to_type)
    elif isinstance(arg, runtime_pandas_types):
        return arg.values.astype(cast_to_type)
    else:
        raise TypeError("Cannot cast {} to {}".format(type(arg), cast_to_type))


def cast_tensor_arg_to_double(arg: tensor_arg_types) -> tensor_arg_types:
    return __cast_to(arg, float)


def cast_tensor_arg_to_integer(arg: tensor_arg_types) -> tensor_arg_types:
    return __cast_to(arg, int)


def cast_tensor_arg_to_boolean(arg: tensor_arg_types) -> tensor_arg_types:
    return __cast_to(arg, bool)
