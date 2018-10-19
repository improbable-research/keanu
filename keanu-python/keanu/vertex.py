import numpy as np
import numbers

import keanu as kn
from keanu.context import KeanuContext
from keanu.base import JavaObjectWrapper


context = KeanuContext()


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
        return kn.generated.vertex.Addition(self, other)

    def __radd__(self, other):
        return kn.generated.vertex.Addition(other, self)

    def __sub__(self, other):
        return kn.generated.vertex.Difference(self, other)

    def __rsub__(self, other):
        return kn.generated.vertex.Difference(other, self)

    def __mul__(self, other):
        return kn.generated.vertex.Multiplication(self, other)

    def __rmul__(self, other):
        return kn.generated.vertex.Multiplication(other, self)

    def __pow__(self, other):
        return kn.generated.vertex.Power(self, other)

    def __rpow__(self, other):
        return kn.generated.vertex.Power(other, self)
        
    def __truediv__(self, other):
        return kn.generated.vertex.Division(self, other)

    def __rtruediv__(self, other):
        return kn.generated.vertex.Division(other, self)

    def __eq__(self, other):
        return kn.generated.vertex.Equals(self, other)
        
    def __req__(self, other):
        return kn.generated.vertex.Equals(self, other)

    def __ne__(self, other):
        return kn.generated.vertex.NotEquals(self, other)
        
    def __rne__(self, other):
        return kn.generated.vertex.NotEquals(self, other)

    def __gt__(self, other):
        return kn.generated.vertex.GreaterThan(self, other)

    def __ge__(self, other):
        return kn.generated.vertex.GreaterThanOrEqual(self, other)

    def __lt__(self, other):
        return kn.generated.vertex.LessThan(self, other)

    def __le__(self, other):
        return kn.generated.vertex.LessThanOrEqual(self, other)

class Vertex(JavaObjectWrapper, VertexOps):
    def __init__(self, ctor, *args):
        super(Vertex, self).__init__(ctor, *(Vertex.__parse_args(*args)))

    def observe(self, v):
        from keanu.const import Tensor
        self.unwrap().observe(Tensor(v).unwrap())

    @staticmethod
    def __parse_args(args):
        return list(map(Vertex.__parse_arg, args))

    @staticmethod
    def __parse_arg(arg):
        if isinstance(arg, np.ndarray):
            return kn.Const(arg).unwrap()
        elif isinstance(arg, numbers.Number):
            return Vertex.__parse_arg(np.array([[arg]]))
        elif isinstance(arg, JavaObjectWrapper):
            return arg.unwrap()
        elif isinstance(arg, list) and all(isinstance(x, numbers.Number) for x in arg):
            return context.to_java_long_array(arg)
        else:
            raise ValueError("Can't parse generic argument. Was given {}".format(type(arg)))
