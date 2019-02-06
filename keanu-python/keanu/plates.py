from typing import Callable, Generator, Dict, Optional

from py4j.java_gateway import java_import
from py4j.protocol import Py4JJavaError

from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.functional import Consumer
from keanu.vertex import Vertex, cast_to_double_vertex, vertex_constructor_param_types
from keanu.vertex.label import _VertexLabel

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.plating.Plates")
java_import(k.jvm_view(), "io.improbable.keanu.plating.PlateBuilder")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.SimpleVertexDictionary")

class Plate(JavaObjectWrapper):

    def add(self, vertex: Vertex, label: Optional[str] = None) -> None:
        if label is None:
            self.unwrap().add(vertex.unwrap())
        else:
            self.unwrap().add(_VertexLabel.create_maybe_with_namespace(label).unwrap(), vertex.unwrap())

    def get(self, label: str) -> Vertex:
        return Vertex._from_java_vertex(self.unwrap().get(_VertexLabel.create_maybe_with_namespace(label).unwrap()))


class Plates(JavaObjectWrapper):

    def __init__(self,
                 count: int,
                 factory: Callable[[Plate], None],
                 initial_state: Dict[str, vertex_constructor_param_types] = None
                 ):
        consumer = Consumer(lambda p: factory(Plate(p)))
        builder = k.jvm_view().PlateBuilder()

        if initial_state is not None:
            initial_state_java = k.to_java_map({_VertexLabel(k): cast_to_double_vertex(v).unwrap() for (k, v) in initial_state.items()})
            vertex_dictionary = k.jvm_view().SimpleVertexDictionary.backedBy(initial_state_java)
            builder = builder.withInitialState(vertex_dictionary)

        builder = builder.count(count)
        builder = builder.withFactory(consumer)


        plates = builder.build()
        super(Plates, self).__init__(plates)


    def __iter__(self) -> Generator[Plate, None, None]:
        iterator = self.unwrap().iterator()
        while iterator.hasNext():
            yield Plate(iterator.next())

    def size(self) -> int:
        return self.unwrap().size()


    @staticmethod
    def proxy_for(label: str) -> str:
        proxy_label = k.jvm_view().PlateBuilder.proxyFor(_VertexLabel(label).unwrap())
        return proxy_label.getQualifiedName()
