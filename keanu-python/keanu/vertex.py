import numpy as np
import numbers

from keanu.context import KeanuContext
from keanu.base import JavaObjectWrapper
from keanu.const import Const


context = KeanuContext()


class VertexOps:
    def __gt__(self, other):
        from keanu.generated.vertex import GreaterThan
        return GreaterThan(self, other)


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
            return Const(arg).unwrap()
        elif isinstance(arg, numbers.Number):
            return arg
        elif isinstance(arg, JavaObjectWrapper):
            return arg.unwrap()
        elif isinstance(arg, list) and all(isinstance(x, numbers.Number) for x in arg):
            return context.to_java_long_array(arg)
        else:
            raise ValueError("Can't parse generic argument. Was given {}".format(type(arg)))
