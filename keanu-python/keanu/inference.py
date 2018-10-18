from py4j.java_gateway import java_import
from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext

k = KeanuContext().jvm_view()

java_import(k, "io.improbable.keanu.network.BayesianNetwork")
java_import(k, "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")

class BayesNet(JavaObjectWrapper):
    def __init__(self, vertices):
        self.vertices = vertices
        super(BayesNet, self).__init__(k.BayesianNetwork, vertices)


class InferenceAlgorithm:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def get_posterior_samples(self, net, vertices, sample_count):
        return self.algorithm.withDefaultConfig().getPosteriorSamples(
            net.unwrap(),
            vertices,
            sample_count)


class MetropolisHastings(InferenceAlgorithm):
    def __init__(self):
        super(MetropolisHastings, self).__init__(k.MetropolisHastings)
