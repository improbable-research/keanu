from py4j.java_gateway import java_import
from .context import KeanuContext
from .base import JavaObjectWrapper
from typing import Optional

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.KeanuRandom")
java_import(k.jvm_view(), "io.improbable.keanu.vertices.VertexId")


class KeanuRandom(JavaObjectWrapper):

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is None:
            super(KeanuRandom, self).__init__(k.jvm_view().KeanuRandom())
        else:
            super(KeanuRandom, self).__init__(k.jvm_view().KeanuRandom(seed))

    def next_double(self) -> float:
        return self.unwrap().nextDouble()

    @staticmethod
    def set_default_random_seed(seed: int) -> None:
        k.jvm_view().KeanuRandom.setDefaultRandomSeed(seed)


def set_deterministic_state() -> None:
    KeanuRandom.set_default_random_seed(0)
    k.jvm_view().VertexId.resetIdGenerator()
