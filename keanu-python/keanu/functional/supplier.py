from typing import Callable

from py4j.java_gateway import JavaObject


class Supplier:

    def __init__(self, lambda_function: Callable):
        self.lambda_function = lambda_function

    def get(self) -> JavaObject:
        """
        >>> f = Supplier(lambda : "foo")
        >>> f.get()
        'foo'
        """
        return self.lambda_function()

    class Java:
        implements = ["java.util.function.Supplier"]
