from typing import Callable, Any

from py4j.java_gateway import JavaObject

from keanu.functional.hash_shortener import shorten_hash


class BiConsumer:

    def __init__(self, lambda_function: Callable[[JavaObject, JavaObject], None]) -> None:
        self.lambda_function = lambda_function

    def accept(self, arg1: JavaObject, arg2: JavaObject) -> None:
        """
        >>> c = BiConsumer(lambda x,y : print(x + y))
        >>> c.accept("foo", "bar")
        foobar
        """
        self.lambda_function(arg1, arg2)

    def hashCode(self) -> int:
        return shorten_hash(hash(self.lambda_function))

    class Java:
        implements = ["java.util.function.BiConsumer"]
