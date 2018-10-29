import warnings
from keanu.utils import _get_java_name, _get_python_name

class JavaObjectWrapper:
    def __init__(self, val):
        self._val = val
        self._class = self.unwrap().getClass().getSimpleName()

    def __repr__(self):
        return "[{0} => {1}]".format(self._class, type(self))

    def __getattr__(self, k):
        python_name = _get_python_name(k)

        if k != python_name:
            if python_name in self.__class__.__dict__:
                raise AttributeError("{} has no attribute {}. Did you mean {}?".format(self.__class__, k, python_name))

            raise AttributeError("{} has no attribute {}".format(self.__class__, k))

        warnings.warn("\"{}\" is not implemented so Java API was called directly instead".format(k))
        return self.unwrap().__getattr__(_get_java_name(k))

    def unwrap(self):
        return self._val
