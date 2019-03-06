from typing import Callable
from py4j.java_gateway import JavaObject


class JavaObjectWrapper:

    def __init__(self, val: JavaObject) -> None:
        self._val = val
        self._class = self.unwrap().getClass().getSimpleName()

    def __repr__(self) -> str:
        return "[{0} => {1}]".format(self._class, type(self))

    def __getattr__(self, k: str) -> Callable:
        self.__check_if_constructed_without_error(k)
        self.__check_if_unwrapped(k)
        raise AttributeError("{} has no attribute {}".format(self.__class__, k))

    def __check_if_constructed_without_error(self, k: str) -> None:
        if k in ("_val", "_class"):
            raise ValueError("Object did not get properly constructed - probably due to an earlier unhandled Error.")

    def __check_if_unwrapped(self, k: str) -> None:
        # better error message for when JavaObjectWrapper is passed to a method that expects JavaObject
        # see: https://www.py4j.org/advanced_topics.html#converting-python-collections-to-java-collections
        if k == "_get_object_id":
            raise TypeError(
                "Trying to pass {} to a method that expects a JavaObject - did you forget to call unwrap()?".format(
                    self.__class__))

    def unwrap(self) -> JavaObject:
        return self._val
