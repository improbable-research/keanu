from typing import Set, Iterable, Union

from py4j.java_gateway import java_import

from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.vertex import Vertex

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.mcmc.proposal.AcceptanceRateTracker")

proposal_listener_types = Union['AcceptanceRateTracker']


class AcceptanceRateTracker(JavaObjectWrapper):

    def __init__(self) -> None:
        super(AcceptanceRateTracker, self).__init__(k.jvm_view().AcceptanceRateTracker())

    def get_acceptance_rate(self, vertex: Vertex) -> float:
        return self.unwrap().getAcceptanceRate(vertex.unwrap().getId())
