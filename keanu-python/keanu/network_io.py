from py4j.java_gateway import java_import
from py4j.java_gateway import JavaObject
from .context import KeanuContext
from .base import JavaObjectWrapper
from .net import BayesNet
from typing import Dict, Optional, Union, Iterable
from .vertex.base import Vertex
import collections

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.util.io.ProtobufSaver")
java_import(k.jvm_view(), "io.improbable.keanu.util.io.DotSaver")
java_import(k.jvm_view(), "io.improbable.keanu.util.io.JsonSaver")
java_import(k.jvm_view(), "io.improbable.keanu.util.io.MIRSaver")
java_import(k.jvm_view(), "io.improbable.keanu.util.io.ProtobufLoader")
java_import(k.jvm_view(), "io.improbable.keanu.util.io.JsonLoader")
java_import(k.jvm_view(), "io.improbable.keanu.util.io.MIRLoader")
java_import(k.jvm_view(), "java.io.FileOutputStream")
java_import(k.jvm_view(), "java.io.FileInputStream")


class NetworkSaver(JavaObjectWrapper):

    def __init__(self, saver_object: JavaObject):
        super().__init__(saver_object)

    def save(self, filename: str, save_values: bool = False, metadata: Optional[Dict[str, str]] = None):
        output_stream = k.jvm_view().FileOutputStream(filename)
        if metadata is None:
            self.unwrap().save(output_stream, save_values)
        else:
            self.unwrap().save(output_stream, save_values, k.to_java_map(metadata))


class ProtobufSaver(NetworkSaver):

    def __init__(self, net: BayesNet):
        super().__init__(k.jvm_view().ProtobufSaver(net.unwrap()))


class DotSaver(NetworkSaver):

    def __init__(self, net_or_vertices: Union[BayesNet, Iterable[Vertex]]):
        if isinstance(net_or_vertices, BayesNet):
            val = net_or_vertices.unwrap()
        elif isinstance(net_or_vertices, collections.Iterable):
            val = k.to_java_object_set(net_or_vertices)
        else:
            raise TypeError("DotSaver only takes BayesNet or a list of vertices.")

        super().__init__(k.jvm_view().DotSaver(val))


class JsonSaver(NetworkSaver):

    def __init__(self, net: BayesNet):
        super().__init__(k.jvm_view().JsonSaver(net.unwrap()))


class MIRSaver(NetworkSaver):

    def __init__(self, net: BayesNet):
        super().__init__(k.jvm_view().MIRSaver(net.unwrap()))


class NetworkLoader(JavaObjectWrapper):

    def __init__(self, loader_object: JavaObject):
        super().__init__(loader_object)

    def load(self, filename: str) -> BayesNet:
        input_stream = k.jvm_view().FileInputStream(filename)
        read_net = self.unwrap().loadNetwork(input_stream)
        return BayesNet(read_net.getAllVertices())


class ProtobufLoader(NetworkLoader):

    def __init__(self) -> None:
        super().__init__(k.jvm_view().ProtobufLoader())


class JsonLoader(NetworkLoader):

    def __init__(self) -> None:
        super().__init__(k.jvm_view().JsonLoader())


class MIRLoader(NetworkLoader):

    def __init__(self) -> None:
        super().__init__(k.jvm_view().MIRLoader())
