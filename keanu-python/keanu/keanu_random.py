from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from keanu.base import JavaObjectWrapper

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.vertices.dbl.KeanuRandom")


class KeanuRandom(JavaObjectWrapper):
    def __init__(self, seed=None):
        if seed is None:
            super(KeanuRandom, self).__init__(k.jvm_view().KeanuRandom())
        else:
            super(KeanuRandom, self).__init__(k.jvm_view().KeanuRandom(seed))

    def set_default_random_seed(self, seed):
    	k.jvm_view().KeanuRandom.setDefaultRandomSeed(seed)
