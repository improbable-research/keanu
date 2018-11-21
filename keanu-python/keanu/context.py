import sys
import io
import os
import logging
from py4j.java_gateway import JavaGateway, CallbackServerParameters, JavaObject
from typing import Dict, Any, Iterable, List

PATH = os.path.abspath(os.path.dirname(__file__))
ND4J_CLASSPATH_ENVIRONMENT_VARIABLE = "KEANU_ND4J_CLASSPATH"


# python singleton implementation https://stackoverflow.com/a/6798042/741789
class Singleton(type):
    _instances = {}  # type: Dict[KeanuContext, KeanuContext]

    def __call__(cls: Any, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class KeanuContext(metaclass=Singleton):

    def __init__(self) -> None:
        stderr = self.__stderr_with_redirect_disabled_for_jupyter()
        classpath = self.__build_classpath()

        logging.getLogger("keanu").debug("Initiating Py4J gateway with classpath %s" % classpath)

        self._gateway = JavaGateway.launch_gateway(
            classpath=classpath, die_on_exit=True, redirect_stdout=sys.stdout, redirect_stderr=stderr)

        self.__get_random_port_for_callback_server()

        self.__jvm_view = self._gateway.new_jvm_view()

    def __build_classpath(self) -> str:
        keanu_path = os.path.join(PATH, "classpath", "*")
        nd4j_path = os.environ.get(ND4J_CLASSPATH_ENVIRONMENT_VARIABLE)
        if nd4j_path is None:
            return keanu_path
        else:
            return os.pathsep.join([keanu_path, os.path.join(nd4j_path, "*")])

    def __stderr_with_redirect_disabled_for_jupyter(self) -> Any:
        try:
            sys.stderr.fileno()
            return sys.stderr
        except io.UnsupportedOperation:
            return None

    def __get_random_port_for_callback_server(self) -> None:
        # See: https://github.com/bartdag/py4j/issues/147
        self._gateway.start_callback_server(
            CallbackServerParameters(port=0, daemonize=True, daemonize_connections=True))
        jgws = JavaObject("GATEWAY_SERVER", self._gateway._gateway_client)
        jgws.resetCallbackClient(jgws.getCallbackClient().getAddress(),
                                 self._gateway.get_callback_server().get_listening_port())

    def jvm_view(self) -> Any:
        return self.__jvm_view

    def to_java_object_list(self, l: Iterable[Any]) -> Any:
        lst = self._gateway.jvm.java.util.ArrayList()

        for o in l:
            lst.add(o.unwrap())

        return lst

    def to_java_array(self, l: List[Any], klass: Any = None) -> Any:
        if klass is None:
            klass = self.__infer_class_from_array(l)
        array = self._gateway.new_array(klass, len(l))

        for idx, o in enumerate(l):
            array[idx] = o

        return array

    def to_java_long_array(self, l: Any) -> Any:
        return self.to_java_array(l, self._gateway.jvm.long)

    def __infer_class_from_array(self, l: List[Any]) -> Any:
        if len(l) == 0:
            raise ValueError("Cannot infer type because array is empty")

        if isinstance(l[0], bool):
            return self._gateway.jvm.boolean
        elif isinstance(l[0], int):
            return self._gateway.jvm.int
        elif isinstance(l[0], float):
            return self._gateway.jvm.double
        else:
            raise NotImplementedError(
                "Cannot infer class from array because it doesn't contain primitives. Was given {}".format(type(l[0])))
