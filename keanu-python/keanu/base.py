import warnings
from keanu.utils import get_java_name

class JavaObjectWrapper:
    def __init__(self, val):
        self._val = val
        self._class = self.unwrap().getClass().getSimpleName()

    def __repr__(self):
        return "[{0} => {1}]".format(self._class, type(self))

    def __getattr__(self, k):
        java_name = get_java_name(k)

        if not k.islower() and k == java_name:
            raise AttributeError("{} has no attribute {}".format(self, k))

        warnings.warn("\"{}\" is not implemented so Java API was called directly instead".format(k))
        return self.unwrap().__getattr__(java_name)

    def unwrap(self):
        return self._val
