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

    def __getattr__(self, attr: str) -> Callable:
        self.__check_attr_is_snake_case(attr)

        # AttributeError MUST be thrown here when java API does not implement given attr.
        # Otherwise calls to this object may result in unexpected behaviours (e.g. with logic that depends on hasattr)
        java_name = _to_camel_case_name(attr)
        self.__check_java_object_has_attr(attr, java_name)

        logging.getLogger("keanu").warning(
                "\"{}\" is not implemented so Java API \"{}\" was called directly instead".format(attr, java_name))
        return self.unwrap().__getattr__(java_name)

    def unwrap(self) -> JavaObject:
        return self._val

    def __check_attr_is_snake_case(self, attr: str) -> None:
        snake_case = _to_snake_case_name(attr)
        if attr != snake_case:
            if snake_case in self.__class__.__dict__:
                raise AttributeError("{} has no attribute {}. Did you mean {}?".format(self.__class__, attr, snake_case))
            else:
                raise AttributeError("{} has no attribute {}.".format(self.__class__, attr))

    def __check_java_object_has_attr(self, attr: str, java_name: str) -> None:
        if not java_name in dir(self.unwrap()):
            raise AttributeError("{} has no attribute {}.".format(self.__class__, attr))
