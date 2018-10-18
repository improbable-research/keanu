from py4j.java_gateway import java_import
from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.inference import BayesNet
from keanu.vertex import Vertex

k = KeanuContext().jvm_view()

java_import(k, "io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer")
java_import(k, "io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer")

class GradientOptimizer(JavaObjectWrapper):
    def __init__(self, net):
        if isinstance(net, BayesNet):
            super(GradientOptimizer, self).__init__(k.GradientOptimizer.of, net.unwrap())
        elif isinstance(net, Vertex):
            super(GradientOptimizer, self).__init__(k.GradientOptimizer.of, net.unwrap().getConnectedGraph())
        else:
            raise NotImplementedError("Provide a Vertex or BayesNet to the optimizer")

    def max_a_posteriori(self):
        return self.maxAPosteriori()

    def max_likelihood(self):
        return self.maxLikelihood()

class NonGradientOptimizer(JavaObjectWrapper):
    def __init__(self, net):
        if isinstance(net, BayesNet):
            super(NonGradientOptimizer, self).__init__(k.NonGradientOptimizer.of, net.unwrap())
        elif isinstance(net, Vertex):
            super(NonGradientOptimizer, self).__init__(k.NonGradientOptimizer.of, net.unwrap().getConnectedGraph())
        else:
            raise NotImplementedError("Provide a Vertex or BayesNet to the optimizer")

    def max_a_posteriori(self):
        return self.maxAPosteriori()

    def max_likelihood(self):
        return self.maxLikelihood()