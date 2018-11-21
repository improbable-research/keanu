import logging
from .case_conversion import _to_camel_case_name, _to_snake_case_name
from typing import Any, Callable


class JavaObjectWrapper:

    def __init__(self, val: Any) -> None:
        self._val = val
        self._class = self.unwrap().getClass().getSimpleName()

    def __repr__(self) -> str:
        return "[{0} => {1}]".format(self._class, type(self))

    def __getattr__(self, k: str) -> Callable:
        python_name = _to_snake_case_name(k)

        if k != python_name:
            if python_name in self.__class__.__dict__:
                raise AttributeError("{} has no attribute {}. Did you mean {}?".format(self.__class__, k, python_name))

            raise AttributeError("{} has no attribute {}".format(self.__class__, k))

        java_name = _to_camel_case_name(k)
        logging.warning("\"{}\" is not implemented so Java API \"{}\" was called directly instead".format(k, java_name))
        return self.unwrap().__getattr__(java_name)

    def unwrap(self) -> Any:
        return self._val
