class JavaObjectWrapper:
    def __init__(self, val):
        self._val = val
        self._class = self.unwrap().getClass().getSimpleName()

    def __repr__(self):
        return "[{0} => {1}]".format(self._class, type(self))

    def __getattr__(self, k):
        py_method = JavaObjectWrapper.__get_py_name(k)
        if py_method in self.__dict__:
            return self.__dict__[py_method]
        return self.unwrap().__getattr__(JavaObjectWrapper.__get_java_name(k))

    def unwrap(self):
        return self._val

    @staticmethod
    def __get_java_name(column):
       first, *rest = column.split('_')
       return first + ''.join(word.capitalize() for word in rest)

    @staticmethod
    def __get_py_name(s):
        import re
        _underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
        _underscorer2 = re.compile('([a-z0-9])([A-Z])')
        subbed = _underscorer1.sub(r'\1_\2', s)
        return _underscorer2.sub(r'\1_\2', subbed).lower()


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

    def contains(self, py_element):
        return self.unwrap().contains(py_element.unwrap())

    def get(self, index):
        return JavaObjectWrapper(self.unwrap().get(index))


class JavaSet(JavaObjectWrapper):
    def __init__(self, java_set):
        super(JavaSet, self).__init__(java_set)

    def contains(self, py_element):
        return self.unwrap().contains(py_element.unwrap())


class UnaryLambda:
    def __init__(self, fn):
        self._fn = fn

    def apply(self, val):
        return self._fn(val)

    class Java:
        implements = ["java.util.function.Function"]

