import sys
import io
import os.path
from py4j.java_gateway import JavaGateway, JavaObject, CallbackServerParameters

PATH = os.path.abspath(os.path.dirname(__file__))
CLASSPATH = os.path.join(PATH, "keanu-python-all.jar")


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class KeanuContext(metaclass=Singleton):
    def __init__(self):
        stderr = sys.stderr

        # Disable redirecting of stderr in jupyter notebooks as it doesn't work
        try:
            sys.stderr.fileno()
        except io.UnsupportedOperation:
            stderr = None

        self._gateway = JavaGateway.launch_gateway(
            classpath=CLASSPATH,
            die_on_exit=True,
            redirect_stdout=sys.stdout,
            redirect_stderr=stderr
        )

        # Get random port for callback server
        # See: https://github.com/bartdag/py4j/issues/147
        self._gateway.start_callback_server(CallbackServerParameters(port=0, daemonize=True, daemonize_connections=True))
        jgws = JavaObject("GATEWAY_SERVER", self._gateway._gateway_client)
        jgws.resetCallbackClient(jgws.getCallbackClient().getAddress(), self._gateway.get_callback_server().get_listening_port())

    def jvm_view(self):
        return self._gateway.new_jvm_view()

    def list_vertex_classes(self):
        return self._gateway.jvm.io.improbable.keanu.python.Keanu().getVertexClasses()

    def _infer_class_from_array(self, l):
        if len(l) == 0:
            raise ValueError("Cannot infer type because array is empty")

        if isinstance(l[0], int):
            return self._gateway.jvm.int
        elif isinstance(l[0], float):
            return self._gateway.jvm.double
        elif isinstance(l[0], bool):
            return self._gateway.jvm.boolean
        else:
            return self._gateway.jvm.Object

    def to_java_array(self, l, klass=None, cast=lambda x: x):
        if klass is None:
            klass = self._infer_class_from_array(l)
        array = self._gateway.new_array(klass, len(l))

        for idx, o in enumerate(l):
            array[idx] = cast(o)

        return array
