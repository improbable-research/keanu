import numpy as np
import collections
import numbers

import keanu as kn
from keanu.context import KeanuContext
from keanu.base import JavaObjectWrapper
from keanu.tensor import Tensor
from keanu.vartypes import primitive_types, const_arg_types

k = KeanuContext()


class VertexOps:
    """
    __array_ufunc__ is a NumPy thing that enables you to intercept and handle the numpy operation.
    Without this the right operators would fail.
    See https://docs.scipy.org/doc/numpy-1.13.0/neps/ufunc-overrides.html
    """
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        methods = {
            "equal" : VertexOps.__eq__,
            "not_equal" : VertexOps.__ne__,
            "add" : VertexOps.__radd__,
            "subtract" : VertexOps.__rsub__,
            "multiply" : VertexOps.__rmul__,
            "power" : VertexOps.__rpow__,
            "true_divide" : VertexOps.__rtruediv__,
            "floor_divide" : VertexOps.__rfloordiv__,
            "greater" : VertexOps.__lt__,
            "greater_equal" : VertexOps.__le__,
            "less" : VertexOps.__gt__,
            "less_equal" : VertexOps.__ge__,
        }
        if method == "__call__":
            try:
                dispatch_method = methods[ufunc.__name__]
                return dispatch_method(inputs[1], inputs[0])
            except KeyError:
                raise NotImplementedError("NumPy ufunc of type %s not implemented" % ufunc.__name__)
        else:
            raise NotImplementedError("NumPy ufunc method %s not implemented" % method)

    def __add__(self, other):
        return kn.vertex.generated.Addition(self, other)

    def __radd__(self, other):
        return kn.vertex.generated.Addition(other, self)

    def __sub__(self, other):
        return kn.vertex.generated.Difference(self, other)

    def __rsub__(self, other):
        return kn.vertex.generated.Difference(other, self)

    def __mul__(self, other):
        return kn.vertex.generated.Multiplication(self, other)

    def __rmul__(self, other):
        return kn.vertex.generated.Multiplication(other, self)

    def __pow__(self, other):
        return kn.vertex.generated.Power(self, other)

    def __rpow__(self, other):
        return kn.vertex.generated.Power(other, self)

    def __truediv__(self, other):
        return kn.vertex.generated.Division(self, other)

    def __rtruediv__(self, other):
        return kn.vertex.generated.Division(other, self)

    def __floordiv__(self, other):
        return kn.vertex.generated.IntegerDivision(self, other)

    def __rfloordiv__(self, other):
        return kn.vertex.generated.IntegerDivision(other, self)

    def __eq__(self, other):
        return kn.vertex.generated.Equals(self, other)

    def __req__(self, other):
        return kn.vertex.generated.Equals(self, other)

    def __ne__(self, other):
        return kn.vertex.generated.NotEquals(self, other)

    def __rne__(self, other):
        return kn.vertex.generated.NotEquals(self, other)

    def __gt__(self, other):
        return kn.vertex.generated.GreaterThan(self, other)

    def __ge__(self, other):
        return kn.vertex.generated.GreaterThanOrEqual(self, other)

    def __lt__(self, other):
        return kn.vertex.generated.LessThan(self, other)

    def __le__(self, other):
        return kn.vertex.generated.LessThanOrEqual(self, other)

    def __abs__(self):
        return kn.vertex.generated.Abs(self)

    def __round__(self):
        return kn.vertex.generated.Round(self)

    def __floor__(self):
        return kn.vertex.generated.Floor(self)

    def __ceil__(self):
        return kn.vertex.generated.Ceil(self)


class Vertex(JavaObjectWrapper, VertexOps):
    def __init__(self, val, *args):
        if args:
            ctor = val
            val = ctor(*(Vertex.__parse_args(args)))

        super(Vertex, self).__init__(val)

    def __hash__(self):
        return hash(self.get_id())

    def observe(self, v):
        self.unwrap().observe(Tensor(v).unwrap())

    def set_value(self, v):
        self.unwrap().setValue(Tensor(v).unwrap())

    def set_and_cascade(self, v):
        self.unwrap().setAndCascade(Tensor(v).unwrap())

    def sample(self):
        return Tensor._to_ndarray(self.unwrap().sample())

    def get_value(self):
        return Tensor._to_ndarray(self.unwrap().getValue())

    def get_connected_graph(self):
        return Vertex._to_generator(self.unwrap().getConnectedGraph())

    def get_id(self):
        return Vertex._get_python_id(self.unwrap())

    @staticmethod
    def __parse_args(args):
        return list(map(Vertex.__parse_arg, args))

    @staticmethod
    def __parse_arg(arg):
        if isinstance(arg, const_arg_types):
            return kn.vertex.const.Const(arg).unwrap()
        elif isinstance(arg, JavaObjectWrapper):
            return arg.unwrap()
        elif isinstance(arg, collections.Iterable) and all(isinstance(x, primitive_types) for x in arg):
            return k.to_java_long_array(arg)
        else:
            raise ValueError("Can't parse generic argument. Was given {}".format(type(arg)))

    @staticmethod
    def _to_generator(java_vertices):
        return (Vertex(java_vertex) for java_vertex in java_vertices)

    @staticmethod
    def _get_python_id(java_vertex):
        return tuple(java_vertex.getId().getValue())
