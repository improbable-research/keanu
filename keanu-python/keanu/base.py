import warnings
from keanu.utils import get_java_name, get_python_name

class JavaObjectWrapper:
    def __init__(self, val):
        self._val = val
        self._class = self.unwrap().getClass().getSimpleName()

    def __repr__(self):
        return "[{0} => {1}]".format(self._class, type(self))

    def __getattr__(self, k):
        python_name = get_python_name(k)
        if python_name in self.__class__.__dict__:
            return getattr(self, python_name)

        java_name = get_java_name(k)
        warnings.warn("\"{}\" is not implemented so Java API was called directly instead".format(k))
        return self.unwrap().__getattr__(java_name)

    def unwrap(self):
        return self._val


class JavaCtor(JavaObjectWrapper):
    def __init__(self, ctor, *args):
        super(JavaCtor, self).__init__(ctor(*args))
        self._args = args

    def __repr__(self):
        args = [str(arg) for arg in self._args]
        return "[{0} => {1}: ({2})]".format(self._class, type(self), ",".join(args))


class JavaList(JavaObjectWrapper):
    def __init__(self, java_list):
        super(JavaList, self).__init__(java_list)

class JavaSet(JavaObjectWrapper):
    def __init__(self, java_set):
        super(JavaSet, self).__init__(java_set)
