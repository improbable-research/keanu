from py4j.java_gateway import java_import
from .context import KeanuContext
from .base import JavaObjectWrapper
from .net import BayesNet

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.network.ProtobufSaver")

class ProtobufSaver(JavaObjectWrapper):

    def __init__(self, net: BayesNet):
        super(ProtobufSaver, self).__init__(k.jvm_view().ProtobufSaver(net))