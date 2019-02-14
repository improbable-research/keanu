from typing import Callable, Generator, Dict, Optional, Any, Iterator

from py4j.java_gateway import java_import

from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.functional import Consumer
from keanu.functional import BiConsumer
from keanu.functional import JavaIterator
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
            self.unwrap().add(_VertexLabel(label).unwrap(), vertex.unwrap())

    def get(self, label: str) -> Vertex:
        return Vertex._from_java_vertex(self.unwrap().get(_VertexLabel(label).unwrap()))


class Plates(JavaObjectWrapper):

    def __init__(self,
                 factory: Callable[..., None],
                 count: int = None,
                 data_generator: Iterator[Dict[str, Any]] = None,
                 initial_state: Dict[str, vertex_constructor_param_types] = None):

        builder = k.jvm_view().PlateBuilder()

        if initial_state is not None:
            initial_state_java = k.to_java_map(
                {_VertexLabel(k): cast_to_double_vertex(v).unwrap() for (k, v) in initial_state.items()})
            vertex_dictionary = k.jvm_view().SimpleVertexDictionary.backedBy(initial_state_java)
            builder = builder.withInitialState(vertex_dictionary)

        assert (count is None) ^ (data_generator is None), "You must specify either a count or a data_generator"

        if count is not None:
            function = lambda p: factory(Plate(p))
            consumer = Consumer(function)
            builder = builder.count(count).withFactory(consumer)

        if data_generator is not None:
            bifunction = lambda p, data: factory(Plate(p), data)
            biconsumer = BiConsumer(bifunction)
            data_generator_java = (k.to_java_map(m) for m in data_generator)
            builder = builder.fromIterator(JavaIterator(data_generator_java)).withFactory(biconsumer)

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
        """
        >>> Plates.proxy_for("foo")
        'proxy_for.foo'
        """
        label_java = _VertexLabel(label).unwrap()
        proxy_label_java = k.jvm_view().PlateBuilder.proxyFor(label_java)
        return proxy_label_java.getQualifiedName()
