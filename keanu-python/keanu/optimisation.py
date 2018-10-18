from py4j.java_gateway import java_import
from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.inference import BayesNet
from keanu.vertex import Vertex

k = KeanuContext().jvm_view()

java_import(k, "io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer")
java_import(k, "io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer")


class Optimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def max_a_posteriori(self):
        return self.optimizer.maxAPosteriori()

    def max_likelihood(self):
        return self.optimizer.maxLikelihood()


class GradientOptimizer(Optimizer):
    def __init__(self, net, max_evaluations=None, relative_threshold=None, absolute_threshold=None):
        builder = k.GradientOptimizer.builder()
        if isinstance(net, BayesNet):
            builder = builder.bayesianNetwork(net.unwrap())
        elif isinstance(net, Vertex):
            builder = builder.bayesianNetwork(BayesNet(net.unwrap().getConnectedGraph()).unwrap())
        if max_evaluations is not None:
            builder = builder.maxEvaluations(max_evaluations)
        if relative_threshold is not None:
            builder = builder.relativeThreshold(relative_threshold)
        if absolute_threshold is not None:
            builder = builder.absoluteThreshold(absolute_threshold)

        super(GradientOptimizer, self).__init__(builder.build)


class NonGradientOptimizer(Optimizer):
    def __init__(self, net, max_evaluations=None, bounds_range=None, initial_trust_region_radius=None, stopping_trust_region_radius=None):
        builder = k.NonGradientOptimizer.builder()
        if isinstance(net, BayesNet):
            builder = builder.bayesianNetwork(net.unwrap())
        elif isinstance(net, Vertex):
            builder = builder.bayesianNetwork(BayesNet(net.unwrap().getConnectedGraph()).unwrap())
        if max_evaluations is not None:
            builder = builder.maxEvaluations(max_evaluations)
        if bounds_range is not None:
            builder = builder.boundsRange(bounds_range)
        if initial_trust_region_radius is not None:
            builder = builder.initialTrustRegionRadius(initial_trust_region_radius)
        if stopping_trust_region_radius is not None:
            builder = builder.stoppingTrustRegionRadius(stopping_trust_region_radius)

        super(NonGradientOptimizer, self).__init__(builder.build)