class JavaObjectWrapper:
    def __init__(self, ctor, *args):
        self._ctor = ctor
        self._args = args
        try:
            self._val = ctor(*args)
        except TypeError as e:
            raise EnvironmentError(
                "Cannot call %s - this may be because the keanu jar has not been loaded into py4j's JVM's classpath" 
                % JavaObjectWrapper.print_signature(ctor, args)
            ) from e        
        self._class = self.unwrap().getClass().getSimpleName()

    def __repr__(self):
        return "[{0} => {1}: {2}]".format(self._class, type(self), JavaObjectWrapper.print_signature(self._ctor, self._args))

    @staticmethod
    def print_signature(_ctor, _args):
        args = [str(arg) for arg in _args]
        return "{0}({1})".format(_ctor._fqn, ",".join(args))

    def __getattr__(self, k):
        if k in self.__dict__:
            return self.__dict__[k]
        return self.unwrap().__getattr__(k)

    def unwrap(self):
        return self._val
