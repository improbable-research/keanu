from typing import Callable

from py4j.java_gateway import JavaObject


class Consumer:

    def __init__(self, lambda_function: Callable) -> None:
        self.lambda_function = lambda_function

    def accept(self, arg: JavaObject) -> None:
        """
        >>> c = Consumer(lambda x : print(x))
        >>> c.accept("foo")
        foo
        """
        self.lambda_function(arg)

    class Java:
        implements = ["java.util.function.Consumer"]
