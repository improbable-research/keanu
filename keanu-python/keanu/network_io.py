from py4j.java_gateway import java_import
from .context import KeanuContext
from .base import JavaObjectWrapper
from .net import BayesNet
from typing import Dict, Optional

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.util.io.ProtobufSaver")
java_import(k.jvm_view(), "io.improbable.keanu.util.io.DotSaver")
java_import(k.jvm_view(), "io.improbable.keanu.util.io.JsonSaver")
java_import(k.jvm_view(), "java.io.FileOutputStream")


def _save_network(saver, filename: str, save_values: bool = False, metadata: Optional[Dict[str, str]] = None):
    output_stream = k.jvm_view().FileOutputStream(filename)
    if metadata is None:
        saver.unwrap().save(output_stream, save_values)
    else:
        saver.unwrap().save(output_stream, save_values, k.to_java_map(metadata))


class ProtobufSaver(JavaObjectWrapper):

    def __init__(self, net: BayesNet):
        super(ProtobufSaver, self).__init__(k.jvm_view().ProtobufSaver(net.unwrap()))

    def save(self, filename: str, save_values: bool = False, metadata: Optional[Dict[str, str]] = None):
        _save_network(self, filename, save_values, metadata)


class DotSaver(JavaObjectWrapper):

    def __init__(self, net: BayesNet):
        super(DotSaver, self).__init__(k.jvm_view().DotSaver(net.unwrap()))

    def save(self, filename: str, save_values: bool = False, metadata: Optional[Dict[str, str]] = None):
        _save_network(self, filename, save_values, metadata)


class JsonSaver(JavaObjectWrapper):

    def __init__(self, net: BayesNet):
        super(JsonSaver, self).__init__(k.jvm_view().JsonSaver(net.unwrap()))

    def save(self, filename: str, save_values: bool = False, metadata: Optional[Dict[str, str]] = None):
        _save_network(self, filename, save_values, metadata)
