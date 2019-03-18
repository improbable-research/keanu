import collections
from typing import List, Tuple, Iterator, Union, SupportsRound, Optional, Callable
from typing import cast as typing_cast

import numpy as np
from py4j.java_collections import JavaList, JavaArray
from py4j.java_gateway import JavaObject, JavaMember

import keanu as kn
from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.tensor import Tensor
from keanu.vartypes import (tensor_arg_types, wrapped_java_types, shape_types, numpy_types, runtime_wrapped_java_types,
                            runtime_primitive_types, runtime_numpy_types, runtime_pandas_types, runtime_float_types,
                            runtime_str_types)
from keanu.vertex.label import _VertexLabel

k = KeanuContext()

vertex_operation_param_types = Union['Vertex', tensor_arg_types]
vertex_constructor_param_types = Union['Vertex', tensor_arg_types, wrapped_java_types, str]


class Vertex(JavaObjectWrapper, SupportsRound['Vertex']):

    def __init__(self, val_or_ctor: Union[JavaMember, JavaObject], label: Optional[str],
                 *args: Union[vertex_constructor_param_types, shape_types]) -> None:
        val: JavaObject
        if args:
            ctor = val_or_ctor
            val = ctor(*(Vertex.__parse_args(args)))
        else:
            val = typing_cast(JavaObject, val_or_ctor)

        super(Vertex, self).__init__(val)
        if label is not None:
            self.set_label(label)

    def cast(self, v: tensor_arg_types) -> tensor_arg_types:
        return v

    def __bool__(self) -> bool:
        raise TypeError(
            'Keanu vertices cannot be used as a predicate in a Python "if" statement. Please use keanu.vertex.If instead.'
        )

    def __hash__(self) -> int:
        return hash(self.get_id())

    def observe(self, v: tensor_arg_types) -> None:
        self.unwrap().observe(Tensor(self.cast(v)).unwrap())

    def unobserve(self) -> None:
        self.unwrap().unobserve()

    def set_value(self, v: tensor_arg_types) -> None:
        self.unwrap().setValue(Tensor(self.cast(v)).unwrap())

    def set_and_cascade(self, v: tensor_arg_types) -> None:
        self.unwrap().setAndCascade(Tensor(self.cast(v)).unwrap())

    def set_label(self, label: Optional[str]) -> None:
        if label is None:
            raise ValueError("label cannot be None.")
        self.unwrap().setLabel(_VertexLabel(label).unwrap())

    def sample(self) -> numpy_types:
        return Tensor._to_ndarray(self.unwrap().sample())

    def get_value(self) -> numpy_types:
        return Tensor._to_ndarray(self.unwrap().getValue())

    def iter_connected_graph(self) -> Iterator['Vertex']:
        return Vertex._to_generator(self.unwrap().getConnectedGraph())

    def get_id(self) -> Tuple[int, ...]:
        return Vertex._get_python_id(self.unwrap())

    def get_label(self) -> Optional[str]:
        label = self.unwrap().getLabel()
        return None if label is None else label.getQualifiedName()

    def get_label_without_outer_namespace(self) -> Optional[str]:
        label = self.unwrap().getLabel()
        return None if label is None else label.withoutOuterNamespace().getQualifiedName()

    def iter_parents(self) -> Iterator['Vertex']:
        return Vertex._to_generator(self.unwrap().getParents())

    def iter_children(self) -> Iterator['Vertex']:
        return Vertex._to_generator(self.unwrap().getChildren())

    def is_observed(self) -> bool:
        return self.unwrap().isObserved()

    def has_value(self) -> bool:
        return self.unwrap().hasValue()

    """
    __array_ufunc__ is a NumPy thing that enables you to intercept and handle the numpy operation.
    Without this the right operators would fail.
    See https://docs.scipy.org/doc/numpy-1.13.0/neps/ufunc-overrides.html
    """

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, input0: numpy_types, _: 'Vertex') -> 'Vertex':
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
                result = dispatch_method(self, input0)
                return result
            except KeyError:
                raise NotImplementedError("NumPy ufunc of type %s not implemented" % ufunc.__name__)
        else:
            raise NotImplementedError("NumPy ufunc method %s not implemented" % method)

    def __add__(self, other: vertex_operation_param_types) -> 'Vertex':
        other = cast_to_double_vertex_if_integer_vertex(other)
        return kn.vertex.generated.Addition(self, other)

    def __radd__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Addition(other, self)

    def __sub__(self, other: vertex_operation_param_types) -> 'Vertex':
        other = cast_to_double_vertex_if_integer_vertex(other)
        return kn.vertex.generated.Difference(self, other)

    def __rsub__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Difference(other, self)

    def __mul__(self, other: vertex_operation_param_types) -> 'Vertex':
        other = cast_to_double_vertex_if_integer_vertex(other)
        return kn.vertex.generated.Multiplication(self, other)

    def __rmul__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Multiplication(other, self)

    def __pow__(self, other: vertex_operation_param_types) -> 'Vertex':
        other = cast_to_double_vertex_if_integer_vertex(other)
        return kn.vertex.generated.Power(self, other)

    def __rpow__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Power(other, self)

    def __truediv__(self, other: vertex_operation_param_types) -> 'Vertex':
        other = cast_to_double_vertex_if_integer_vertex(other)
        return kn.vertex.generated.Division(self, other)

    def __rtruediv__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.Division(other, self)

    def __floordiv__(self, other: vertex_operation_param_types) -> 'Vertex':
        other = cast_to_double_vertex_if_integer_vertex(other)
        intermediate = kn.vertex.generated.Division(self, other)

        return kn.vertex.generated.Floor(intermediate)

    def __rfloordiv__(self, other: vertex_operation_param_types) -> 'Vertex':
        intermediate = kn.vertex.generated.Division(other, self)
        return kn.vertex.generated.Floor(intermediate)

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
        elif isinstance(arg, (runtime_primitive_types, JavaObject, runtime_str_types)):
            return arg
        else:
            raise ValueError("Can't parse generic argument. Was given {}".format(type(arg)))

    @staticmethod
    def _from_java_vertex(java_vertex: JavaObject) -> 'Vertex':
        return Vertex(java_vertex, None)

    @staticmethod
    def _to_generator(java_vertices: Union[JavaList, JavaArray]) -> Iterator['Vertex']:
        return (Vertex._from_java_vertex(java_vertex) for java_vertex in java_vertices)

    @staticmethod
    def _get_python_id(java_vertex: JavaObject) -> Tuple[int, ...]:
        return tuple(java_vertex.getId().getValue())


class Double(Vertex):

    def cast(self, v: tensor_arg_types) -> tensor_arg_types:
        return cast_tensor_arg_to_double(v)


class Integer(Vertex):

    def cast(self, v: tensor_arg_types) -> tensor_arg_types:
        return cast_tensor_arg_to_integer(v)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, input0: numpy_types, _: 'Vertex') -> 'Vertex':
        methods = {
            "add": Integer.__radd__,
            "subtract": Integer.__rsub__,
            "multiply": Integer.__rmul__,
            "power": Integer.__rpow__,
            "true_divide": Integer.__rtruediv__,
            "floor_divide": Integer.__rfloordiv__,
        }
        if method == "__call__":
            try:
                dispatch_method = methods[ufunc.__name__]
                result = dispatch_method(self, input0)
                return result
            except KeyError:
                return super().__array_ufunc__(ufunc, method, input0, _)
        else:
            raise NotImplementedError("NumPy ufunc method %s not implemented" % method)

    def __op_based_on_other_type(self, other: vertex_operation_param_types, op: Callable,
                                 integer_op_ctr: Callable) -> 'Vertex':
        if is_floating_type(other):
            # Equivalent to kn.vertex.generated.CastToDouble(self).__add__(other) for add
            return op(kn.vertex.generated.CastToDouble(self))
        else:
            return integer_op_ctr(other)

    def __add__(self, other: vertex_operation_param_types) -> 'Vertex':
        return self.__op_based_on_other_type(
            other, lambda casted_to_double: casted_to_double.__add__(other), lambda other_holder: kn.vertex.generated.
            IntegerAddition(self, other_holder))

    def __radd__(self, other: vertex_operation_param_types) -> 'Vertex':
        return self.__op_based_on_other_type(
            other, lambda casted_to_double: casted_to_double.__radd__(other), lambda other_holder: kn.vertex.generated.
            IntegerAddition(other_holder, self))

    def __sub__(self, other: vertex_operation_param_types) -> 'Vertex':
        return self.__op_based_on_other_type(
            other, lambda casted_to_double: casted_to_double.__sub__(other), lambda other_holder: kn.vertex.generated.
            IntegerDifference(self, other_holder))

    def __rsub__(self, other: vertex_operation_param_types) -> 'Vertex':
        return self.__op_based_on_other_type(
            other, lambda casted_to_double: casted_to_double.__rsub__(other), lambda other_holder: kn.vertex.generated.
            IntegerDifference(other_holder, self))

    def __mul__(self, other: vertex_operation_param_types) -> 'Vertex':
        return self.__op_based_on_other_type(
            other, lambda casted_to_double: casted_to_double.__mul__(other), lambda other_holder: kn.vertex.generated.
            IntegerMultiplication(self, other_holder))

    def __rmul__(self, other: vertex_operation_param_types) -> 'Vertex':
        return self.__op_based_on_other_type(
            other, lambda casted_to_double: casted_to_double.__rmul__(other), lambda other_holder: kn.vertex.generated.
            IntegerMultiplication(other_holder, self))

    def __pow__(self, other: vertex_operation_param_types) -> 'Vertex':
        return self.__op_based_on_other_type(other, lambda casted_to_double: casted_to_double.__pow__(other), lambda
                                             other_holder: kn.vertex.generated.IntegerPower(self, other_holder))

    def __rpow__(self, other: vertex_operation_param_types) -> 'Vertex':
        return self.__op_based_on_other_type(other, lambda casted_to_double: casted_to_double.__rpow__(other), lambda
                                             other_holder: kn.vertex.generated.IntegerPower(other_holder, self))

    def __truediv__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.CastToDouble(self).__truediv__(other)

    def __rtruediv__(self, other: vertex_operation_param_types) -> 'Vertex':
        return kn.vertex.generated.CastToDouble(self).__rtruediv__(other)

    def __floordiv__(self, other: vertex_operation_param_types) -> 'Vertex':
        return self.__op_based_on_other_type(
            other, lambda casted_to_double: casted_to_double.__truediv__(other).__floor__(), lambda other_holder: kn.
            vertex.generated.IntegerDivision(self, other_holder))

    def __rfloordiv__(self, other: vertex_operation_param_types) -> 'Vertex':
        return self.__op_based_on_other_type(
            other, lambda casted_to_double: casted_to_double.__rtruediv__(other).__floor__(), lambda other_holder: kn.
            vertex.generated.IntegerDivision(other_holder, self))


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


def is_floating_type(other: vertex_operation_param_types) -> bool:
    if isinstance(other, np.ndarray):
        return np.issubdtype(other.dtype, np.floating)

    return type(other) == Double or isinstance(other, runtime_float_types)


def cast_to_double_vertex_if_integer_vertex(vertex: vertex_operation_param_types) -> vertex_operation_param_types:
    if type(vertex) == Integer:
        return kn.vertex.generated.CastToDouble(vertex)
    return vertex
