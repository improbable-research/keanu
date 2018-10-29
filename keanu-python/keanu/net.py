from py4j.java_gateway import java_import
from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.vertex import Vertex

context = KeanuContext()
k = context.jvm_view()

java_import(k, "io.improbable.keanu.network.BayesianNetwork")

class BayesNet(JavaObjectWrapper):
    def __init__(self, vertices):
        java_vertices = context.to_java_list([vertex.unwrap() for vertex in vertices])

        super(BayesNet, self).__init__(k.BayesianNetwork(java_vertices))

    def get_latent_or_observed_vertices(self):
        return Vertex._to_python_list(self.unwrap().getLatentOrObservedVertices())

    def get_latent_vertices(self):
        return Vertex._to_python_list(self.unwrap().getLatentVertices())

    def get_observed_vertices(self):
        return Vertex._to_python_list(self.unwrap().getObservedVertices())

    def get_continuous_latent_vertices(self):
        return Vertex._to_python_list(self.unwrap().getContinuousLatentVertices())

    def get_discrete_latent_vertices(self):
        return Vertex._to_python_list(self.unwrap().getDiscreteLatentVertices())

    def probe_for_non_zero_probability(self, attempts, random):
        self.unwrap().probeForNonZeroProbability(attempts, random.unwrap())
