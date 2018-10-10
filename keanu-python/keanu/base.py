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
