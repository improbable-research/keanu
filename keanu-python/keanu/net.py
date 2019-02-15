from py4j.java_gateway import java_import
from .base import JavaObjectWrapper
from .context import KeanuContext
from .vertex.base import Vertex
from .keanu_random import KeanuRandom
from typing import Any, Iterator, Iterable, Optional
from .vertex.label import _VertexLabel

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.network.BayesianNetwork")
java_import(k.jvm_view(), "io.improbable.keanu.network.KeanuProbabilisticModel")
java_import(k.jvm_view(), "io.improbable.keanu.network.KeanuProbabilisticModelWithGradient")


class BayesNet(JavaObjectWrapper):

    def __init__(self, vertices: Iterable[Any]) -> None:
        java_vertices = k.to_java_object_list(vertices)

        super(BayesNet, self).__init__(k.jvm_view().BayesianNetwork(java_vertices))

    def iter_latent_or_observed_vertices(self) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getLatentOrObservedVertices())

    def iter_latent_vertices(self) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getLatentVertices())

    def iter_observed_vertices(self) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getObservedVertices())

    def iter_continuous_latent_vertices(self) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getContinuousLatentVertices())

    def iter_discrete_latent_vertices(self) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getDiscreteLatentVertices())

    def iter_all_vertices(self) -> Iterator[Vertex]:
        return Vertex._to_generator(self.unwrap().getAllVertices())

    def probe_for_non_zero_probability(self, attempts: int, random: KeanuRandom) -> None:
        self.unwrap().probeForNonZeroProbability(attempts, random.unwrap())

    def get_vertex_by_label(self, label: str) -> Optional[Vertex]:
        java_vertex = self.unwrap().getVertexByLabel(_VertexLabel(label).unwrap())
        return Vertex._from_java_vertex(java_vertex) if java_vertex else None


class ProbabilisticModel(JavaObjectWrapper):

    def __init__(self, net: BayesNet) -> None:
        super(ProbabilisticModel, self).__init__(k.jvm_view().KeanuProbabilisticModel(net.unwrap()))


class ProbabilisticModelWithGradient(JavaObjectWrapper):

    def __init__(self, net: BayesNet) -> None:
        super(ProbabilisticModelWithGradient,
              self).__init__(k.jvm_view().KeanuProbabilisticModelWithGradient(net.unwrap()))
