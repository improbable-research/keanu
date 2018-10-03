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
        super(Vertex, self).__init__(ctor, *(self.__parse_args(*args)))

    def observe(self, v):
        from keanu.tensor import Tensor
        self.unwrap().observe(Tensor(v).unwrap())

    def __parse_args(self, args):
        return list(map(self.__parse_arg, args))

    def __parse_arg(self, arg):
        if isinstance(arg, np.ndarray) or isinstance(arg, numbers.Number):
            from keanu.tensor import Const
            return Const(arg).unwrap()
        elif isinstance(arg, JavaObjectWrapper):
            return arg.unwrap()
        elif isinstance(arg, list):
            return context.to_java_array(arg)
        else:
            return arg
