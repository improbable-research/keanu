from keanu.context import KeanuContext
from py4j.java_gateway import java_import
from keanu.vertex import Vertex

k = KeanuContext()
java_import(k.jvm_view(), "io.improbable.keanu.util.*")
java_import(k.jvm_view(), "io.improbable.keanu.util.DescriptionCreator")


class DescriptionCreator:

    @staticmethod
    def create_description(vertex: Vertex) -> str:
        return k.jvm_view().io.improbable.keanu.util.DescriptionCreator.createDescription(vertex.unwrap())
