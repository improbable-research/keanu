from py4j.java_gateway import java_import
from .context import KeanuContext
from .base import JavaObjectWrapper
from .net import BayesNet
from typing import Dict, Optional

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.util.io.ProtobufSaver")
java_import(k.jvm_view(), "java.io.FileOutputStream")

class ProtobufSaver(JavaObjectWrapper):

    def __init__(self, net: BayesNet):
        super(ProtobufSaver, self).__init__(k.jvm_view().ProtobufSaver(net.unwrap()))

    def save(self, filename: str, save_values: bool = False, metadata: Optional[Dict[str, str]] = None):
        output_stream = k.jvm_view().FileOutputStream(filename)
        self.unwrap().save(output_stream, save_values, k.to_java_map(metadata))
