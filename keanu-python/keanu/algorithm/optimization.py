from py4j.java_gateway import java_import
from keanu.context import KeanuContext
from keanu.net import BayesNet
from keanu.vertex.base import Vertex

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer")


class Optimizer:
    def __init__(self, optimizer, net):
        self.optimizer = optimizer
        self.net = net

    def max_a_posteriori(self):
        return self.optimizer.maxAPosteriori()

    def max_likelihood(self):
        return self.optimizer.maxLikelihood()

    @staticmethod
    def _build_bayes_net(builder, net):
        if not (isinstance(net, BayesNet) or isinstance(net, Vertex)):
            raise TypeError("net must be a Vertex or a BayesNet. Was given {}".format(type(net)))
        elif isinstance(net, Vertex):
            net = BayesNet(net.get_connected_graph())
        return builder.bayesianNetwork(net.unwrap()), net


class GradientOptimizer(Optimizer):
    def __init__(self, net, max_evaluations=None, relative_threshold=None, absolute_threshold=None):
        builder = k.jvm_view().GradientOptimizer.builder()
        builder, net = Optimizer._build_bayes_net(builder, net)
        if max_evaluations is not None:
            builder.maxEvaluations(max_evaluations)
        if relative_threshold is not None:
            builder.relativeThreshold(relative_threshold)
        if absolute_threshold is not None:
            builder.absoluteThreshold(absolute_threshold)

        super(GradientOptimizer, self).__init__(builder.build(), net)


class NonGradientOptimizer(Optimizer):
    def __init__(self, net, max_evaluations=None, bounds_range=None, initial_trust_region_radius=None, stopping_trust_region_radius=None):
        builder = k.jvm_view().NonGradientOptimizer.builder()
        builder, net = Optimizer._build_bayes_net(builder, net)
        if max_evaluations is not None:
            builder.maxEvaluations(max_evaluations)
        if bounds_range is not None:
            builder.boundsRange(bounds_range)
        if initial_trust_region_radius is not None:
            builder.initialTrustRegionRadius(initial_trust_region_radius)
        if stopping_trust_region_radius is not None:
            builder.stoppingTrustRegionRadius(stopping_trust_region_radius)

        super(NonGradientOptimizer, self).__init__(builder.build(), net)
