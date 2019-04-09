from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from py4j.java_gateway import java_import
from keanu.vertex import Vertex

k = KeanuContext()
java_import(k.jvm_view(), "io.improbable.keanu.util.*")
java_import(k.jvm_view(), "io.improbable.keanu.util.DescriptionCreator")


class DescriptionCreator(JavaObjectWrapper):

    def __init__(self) -> None:
        super(DescriptionCreator, self).__init__(k.jvm_view().io.improbable.keanu.util.DescriptionCreator())

    def create_description(self, vertex: Vertex) -> str:
        return self.unwrap().createDescription(vertex.unwrap())
