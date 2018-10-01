from py4j.java_gateway import java_import
from keanu.base import KeanuContext, JavaObjectWrapper

k = KeanuContext().jvm_view()

java_import(k, "io.improbable.keanu.network.BayesianNetwork")


class BayesNet(JavaObjectWrapper):
    def __init__(self, vertices):
        super(BayesNet, self).__init__(k.BayesianNetwork, vertices)


java_import(k, "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")
java_import(k, "io.improbable.keanu.algorithms.mcmc.Hamiltonian")
java_import(k, "io.improbable.keanu.algorithms.mcmc.NUTS")


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


class Hamiltonian(InferenceAlgorithm):
    def __init__(self):
        super(Hamiltonian, self).__init__(k.Hamiltonian)


class Nuts(InferenceAlgorithm):
    def __init__(self):
        super(Nuts, self).__init__(k.NUTS)
