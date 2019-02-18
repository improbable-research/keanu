import sys
import io
import os
import logging
import nd4j
from py4j.java_gateway import JavaGateway, CallbackServerParameters, JavaObject, JavaClass, JVMView, java_import
from py4j.protocol import Py4JError
from py4j.java_collections import JavaList, JavaArray, JavaSet, JavaMap
from typing import Dict, Any, Iterable, List, Collection
from _io import TextIOWrapper

PATH = os.path.abspath(os.path.dirname(__file__))


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
        nd4j_path = nd4j.get_classpath()
        return os.pathsep.join([keanu_path, os.path.join(nd4j_path, "*")])

    def __stderr_with_redirect_disabled_for_jupyter(self) -> TextIOWrapper:
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

    def jvm_view(self) -> JVMView:
        return self.__jvm_view

    def to_java_map(self, python_map: Dict[Any, Any]) -> JavaMap:
        m = self._gateway.jvm.java.util.HashMap()

        for (k, v) in python_map.items():
            try:
                new_k = k.unwrap()
            except (AttributeError, Py4JError):
                new_k = k

            try:
                new_v = v.unwrap()
            except (AttributeError, Py4JError):
                new_v = v

            m.put(new_k, new_v)

        return m

    def to_java_object_list(self, l: Iterable[Any]) -> JavaList:
        lst = self._gateway.jvm.java.util.ArrayList()

        for o in l:
            try:
                o = o.unwrap()
            except (AttributeError, Py4JError):
                pass

            lst.add(o)

        return lst

    def to_java_object_set(self, l: Iterable[Any]) -> JavaSet:
        set_ = self._gateway.jvm.java.util.HashSet()

        for o in l:
            set_.add(o.unwrap())

        return set_

    def to_java_array(self, l: Collection[Any], klass: JavaClass = None) -> JavaArray:
        if klass is None:
            klass = self.__infer_class_from_array(l)
        array = self._gateway.new_array(klass, len(l))

        for idx, o in enumerate(l):
            array[idx] = o

        return array

    def to_java_long_array(self, l: Collection[Any]) -> JavaArray:
        return self.to_java_array(l, self._gateway.jvm.long)

    def to_java_int_array(self, l: Collection[Any]) -> JavaArray:
        return self.to_java_array(l, self._gateway.jvm.int)

    def to_java_string_array(self, l: Collection[Any]) -> JavaArray:
        return self.to_java_array(l, self._gateway.jvm.String)

    def to_java_vertex_array(self, l: Collection[Any]) -> JavaArray:
        java_import(self.jvm_view(), "io.improbable.keanu.vertices.Vertex")
        return self.to_java_array(list(map(lambda x: x.unwrap(), l)), self.jvm_view().Vertex)

    def __infer_class_from_array(self, l: Collection[Any]) -> JavaClass:
        if len(l) == 0:
            raise ValueError("Cannot infer type because array is empty")

        first_item = l.__iter__().__next__()
        if isinstance(first_item, bool):
            return self._gateway.jvm.boolean
        elif isinstance(first_item, int):
            return self._gateway.jvm.int
        elif isinstance(first_item, float):
            return self._gateway.jvm.double
        else:
            raise NotImplementedError(
                "Cannot infer class from array because it doesn't contain primitives. Was given {}".format(
                    type(first_item)))
