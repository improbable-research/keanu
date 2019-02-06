from py4j.protocol import Py4JJavaError

from keanu.base import JavaObjectWrapper


class JavaException(JavaObjectWrapper):

    def __init__(self, e: Py4JJavaError):
        super().__init__(e.java_exception)
        self.type: str = e.java_exception.getClass().getName()
        self.message: str = e.java_exception.getMessage()

    def __repr__(self) -> str:
        return "{}: {}".format(self.type, self.message)
