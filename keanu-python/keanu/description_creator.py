from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from py4j.java_gateway import java_import
from keanu.vertex import Vertex

k = KeanuContext()
java_import(k.jvm_view(), "io.improbable.keanu.util.*")
java_import(k.jvm_view(), "io.improbable.keanu.util.DescriptionCreator")

description_creator = k.jvm_view().io.improbable.keanu.util.DescriptionCreator()


def create_description(vertex: Vertex) -> str:
    return description_creator.createDescription(vertex.unwrap())
