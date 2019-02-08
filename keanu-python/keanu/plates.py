from typing import Callable, Generator

from py4j.java_gateway import java_import

from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.functional import Consumer
from keanu.vertex import Vertex
from keanu.vertex.label import _VertexLabel

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.plating.Plates")
java_import(k.jvm_view(), "io.improbable.keanu.plating.PlateBuilder")


class Plate(JavaObjectWrapper):

    def add(self, vertex: Vertex) -> None:
        self.unwrap().add(vertex.unwrap())

    def get(self, label: str) -> Vertex:
        return self.unwrap().get(_VertexLabel(label).unwrap())


class Plates(JavaObjectWrapper):

    def __init__(self, factory: Callable[[Plate], None], count: int):
        consumer = Consumer(lambda p: factory(Plate(p)))
        plates = k.jvm_view().PlateBuilder().count(count).withFactory(consumer).build()
        super(Plates, self).__init__(plates)

    def __iter__(self) -> Generator[Plate, None, None]:
        iterator = self.unwrap().iterator()
        while iterator.hasNext():
            yield Plate(iterator.next())

    def size(self) -> int:
        return self.unwrap().size()
