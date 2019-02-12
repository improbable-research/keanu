from typing import Callable

from py4j.java_gateway import JavaObject


class Function:

    def __init__(self, lambda_function: Callable) -> None:
        self.lambda_function = lambda_function

    def apply(self, arg: JavaObject) -> JavaObject:
        """
        >>> f = Function(lambda x : x + 1)
        >>> f.apply(1)
        2
        """
        return self.lambda_function(arg)

    class Java:
        implements = ["java.util.function.Function"]
