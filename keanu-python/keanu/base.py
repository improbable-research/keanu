import numpy as np
import numbers

from keanu.context import KeanuContext

context = KeanuContext()
k = context.jvm_view()


class JavaObjectWrapper:
    def __init__(self, ctor, *args):
        self._val = ctor(*args)
        self._args = args
        self._class = self.unwrap().getClass().getSimpleName()

    def __repr__(self):
        args = [str(arg) for arg in self._args]
        return "[{0} => {1}: ({2})]".format(self._class, type(self), ",".join(args))

    def __getattr__(self, k):
        if k in self.__dict__:
            return self.__dict__[k]
        return self.unwrap().__getattr__(k)

    def unwrap(self):
        return self._val


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
            from keanu.const import Const
            return Const(arg).unwrap()
        elif isinstance(arg, numbers.Number):
            return arg
        elif isinstance(arg, JavaObjectWrapper):
            return arg.unwrap()
        elif isinstance(arg, list):
            return context.to_java_array(arg)
        else:
            raise ValueError("Can't parse generic argument")
