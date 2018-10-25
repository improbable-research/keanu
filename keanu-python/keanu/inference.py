from py4j.java_gateway import java_import
from keanu.base import JavaObjectWrapper, JavaCtor, JavaList, JavaSet
from keanu.context import KeanuContext
from keanu.impl import UnaryLambda
import numpy as np

context = KeanuContext()
k = context.jvm_view()

java_import(k, "io.improbable.keanu.network.BayesianNetwork")
java_import(k, "io.improbable.keanu.algorithms.mcmc.MetropolisHastings")
java_import(k, "io.improbable.keanu.algorithms.mcmc.NUTS")
java_import(k, "io.improbable.keanu.algorithms.mcmc.Hamiltonian")


class BayesNet(JavaCtor):
    def __init__(self, lst):
        if isinstance(lst, (JavaList, JavaSet)):
            vertices = lst.unwrap()
        elif isinstance(lst, list):
            vertices = context.to_java_list([vertex.unwrap() for vertex in lst])
        else:
            raise TypeError("Expected a list. Was given {}".format(type(lst)))

        super(BayesNet, self).__init__(k.BayesianNetwork, vertices)

    def get_latent_or_observed_vertices(self):
        return JavaList(self.unwrap().getLatentOrObservedVertices())

    def get_latent_vertices(self):
        return JavaList(self.unwrap().getLatentVertices())

    def get_observed_vertices(self):
        return JavaList(self.unwrap().getObservedVertices())

    def get_continuous_latent_vertices(self):
        return JavaList(self.unwrap().getContinuousLatentVertices())

    def get_discrete_latent_vertices(self):
        return JavaList(self.unwrap().getDiscreteLatentVertices())

    def probe_for_non_zero_probability(self, attempts, random):
        self.unwrap().probeForNonZeroProbability(attempts, random.unwrap())


class JavaVertexSamples(JavaObjectWrapper):
    def __init__(self, java_vertex_samples):
        super(JavaVertexSamples, self).__init__(java_vertex_samples)

    def probability(self, fn):
        return self.unwrap().probability(UnaryLambda(fn))

    def get_averages(self):
        keanu_tensor = self.unwrap().getAverages()
        return JavaVertexSamples.__to_np_array(keanu_tensor)

    def get_mode(self):
        keanu_tensor = self.unwrap().getMode()
        return JavaVertexSamples.__to_np_array(keanu_tensor)

    @staticmethod
    def __to_np_array(value):
        np_array = np.array(list(value.asFlatArray()))
        return np_array.reshape(value.getShape())


class JavaNetworkSamples(JavaObjectWrapper):
    def __init__(self, java_network_samples):
        super(JavaNetworkSamples, self).__init__(java_network_samples)

    def get(self, vertex):
        return JavaVertexSamples(self.unwrap().get(vertex.unwrap()))

    def get_double_tensor_samples(self, vertex):
        return JavaVertexSamples(self.unwrap().getDoubleTensorSamples(vertex.unwrap()))

    def get_integer_tensor_samples(self, vertex):
        return JavaVertexSamples(self.unwrap().getIntegerTensorSamples(vertex.unwrap()))

    def drop(self, drop_count):
        return JavaNetworkSamples(self.unwrap().drop(drop_count))

    def down_sample(self, down_sample_interval):
        return JavaNetworkSamples(self.unwrap().downSample(down_sample_interval))

    def probability(self, fn):
        return self.unwrap().probability(UnaryLambda(fn))


class InferenceAlgorithm:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def get_posterior_samples(self, net, lst, sample_count):
        if isinstance(lst, JavaList):
            vertices = lst.unwrap()
        elif isinstance(lst, list):
            vertices = context.to_java_list([vertex.unwrap() for vertex in lst])
        else:
            raise TypeError("Expected a list. Was given {}".format(type(lst)))

        network_samples = self.algorithm.withDefaultConfig().getPosteriorSamples(
            net.unwrap(),
            vertices,
            sample_count)

        return JavaNetworkSamples(network_samples)


class MetropolisHastings(InferenceAlgorithm):
    def __init__(self):
        super(MetropolisHastings, self).__init__(k.MetropolisHastings)


class NUTS(InferenceAlgorithm):
    def __init__(self):
        super(NUTS, self).__init__(k.NUTS)


class Hamiltonian(InferenceAlgorithm):
    def __init__(self):
        super(Hamiltonian, self).__init__(k.Hamiltonian)
