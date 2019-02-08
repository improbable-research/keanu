from py4j.protocol import Py4JJavaError

from keanu.base import JavaObjectWrapper


class JavaException(JavaObjectWrapper, Exception):

    def __init__(self, e: Py4JJavaError):
        super().__init__(e.java_exception)

    def get_class(self) -> str:
        return self.unwrap().getClass().getName()

    def get_message(self) -> str:
        return self.unwrap().getMessage()

    def __repr__(self) -> str:
        return self.unwrap().toString()
