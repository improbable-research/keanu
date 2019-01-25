import logging
from .case_conversion import _to_camel_case_name, _to_snake_case_name
from typing import Callable
from py4j.java_gateway import JavaObject


class JavaObjectWrapper:

    def __init__(self, val: JavaObject) -> None:
        self._val = val
        self._class = self.unwrap().getClass().getSimpleName()

    def __repr__(self) -> str:
        return "[{0} => {1}]".format(self._class, type(self))

    def __getattr__(self, k: str) -> Callable:
        self.__check_if_snake_case(k)
        self.__check_if_unwrapped(k)
        self.__check_if_wrapped_java_object_has_camel_cased_attr(k)

        return self.unwrap().__getattr__(_to_camel_case_name(k))

    def __check_if_snake_case(self, k: str) -> None:
        snake_case_name = _to_snake_case_name(k)
        if k != snake_case_name:
            raise AttributeError("{} has no attribute {}. Did you mean {}?".format(self.__class__, k, snake_case_name))

    def __check_if_unwrapped(self, k: str) -> None:
        if _to_snake_case_name(k) == "_get_object_id":
            raise TypeError("Trying to pass JavaObjectWrapper to a method that expects a JavaObject - did you forget to call unwrap()?")

    def __check_if_wrapped_java_object_has_camel_cased_attr(self, k: str) -> None:
        if not _to_camel_case_name(k) in dir(self.unwrap()):
            raise AttributeError("{} has no attribute {}.".format(self.__class__, k))

    def unwrap(self) -> JavaObject:
        return self._val
