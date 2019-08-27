from typing import Union, Optional, Tuple

from py4j.java_gateway import java_import, JavaObject, JavaClass

from keanu.base import JavaObjectWrapper
from keanu.context import KeanuContext
from keanu.net import BayesNet
from keanu.tensor import Tensor
from keanu.vartypes import numpy_types
from keanu.vertex.base import Vertex

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.variational.optimizer.gradient.ConjugateGradient")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.variational.optimizer.RelativeConvergenceChecker")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.variational.optimizer.AbsoluteConvergenceChecker")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.variational.optimizer.ConvergenceChecker")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.variational.optimizer.gradient.Adam")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer")
java_import(k.jvm_view(), "io.improbable.keanu.algorithms.variational.optimizer.nongradient.BOBYQA")
java_import(k.jvm_view(), "io.improbable.keanu.Keanu")


class _OptimizedResult(JavaObjectWrapper):

    def __init__(self, result_object: JavaObject) -> None:
        super().__init__(result_object)

    def fitness(self) -> float:
        return self.unwrap().getFitness()

    def value_for(self, v: Vertex) -> numpy_types:
        return Tensor._to_ndarray(self.unwrap().getValueFor(v.unwrap().getReference()))


_norm = dict(max_abs='MAX_ABS', l2="L2")


def relative(norm: str, tolerance: float):
    return k.jvm_view().RelativeConvergenceChecker(k.jvm_view().ConvergenceChecker.Norm.valueOf(_norm[norm]), tolerance)


def absolute(norm: str, tolerance: float):
    return k.jvm_view().AbsoluteConvergenceChecker(k.jvm_view().ConvergenceChecker.Norm.valueOf(_norm[norm]), tolerance)


_difference = dict(relative=relative, absolute=absolute)


class ConvergenceChecker(JavaObjectWrapper):
    """Check to determine if optimizer has converged
    Parameters
    ----------
    difference : str
        one of {'absolute', 'relative'}
    norm : srt
        one of {'max_abs', 'l2'}
        max_abs is max(abs(postion - next_position))
        l2 is sqrt(sum((position-next_position)**2))
    tolerance : float
        when the norm strategy is less than this, the optimizer is consider converged and will stop
    """

    def __init__(self, difference: str = 'relative', norm: str = 'max_abs', tolerance: float = 1e-6) -> None:
        super().__init__(_difference[difference](norm, tolerance))


class Optimizer:

    def __init__(self, optimizer: JavaObject, net: Union[BayesNet, Vertex]) -> None:
        self.optimizer = optimizer
        self.net = net

    def max_a_posteriori(self) -> _OptimizedResult:
        return _OptimizedResult(self.optimizer.maxAPosteriori())

    def max_likelihood(self) -> _OptimizedResult:
        return _OptimizedResult(self.optimizer.maxLikelihood())

    @staticmethod
    def _build_bayes_net(factory_class: JavaClass,
                         net: Union[BayesNet, Vertex]) -> Tuple[JavaObject, Union[BayesNet, Vertex]]:

        if not (isinstance(net, BayesNet) or isinstance(net, Vertex)):
            raise TypeError("net must be a Vertex or a BayesNet. Was given {}".format(type(net)))
        elif isinstance(net, Vertex):
            net = BayesNet(net.iter_connected_graph())
        return factory_class.builderFor(net.unwrap()), net


class GradientOptimizer(Optimizer):

    def __init__(self, net: Union[BayesNet, Vertex], algorithm: Optional[JavaObjectWrapper] = None) -> None:
        builder, net = Optimizer._build_bayes_net(k.jvm_view().Keanu.Optimizer.Gradient, net)

        if algorithm is not None:
            builder.algorithm(algorithm.unwrap())

        super(GradientOptimizer, self).__init__(builder.build(), net)


class ConjugateGradient(JavaObjectWrapper):

    def __init__(self,
                 max_evaluations: Optional[int] = None,
                 relative_threshold: Optional[float] = None,
                 absolute_threshold: Optional[float] = None) -> None:

        builder = k.jvm_view().ConjugateGradient.builder()

        if max_evaluations is not None:
            builder.maxEvaluations(max_evaluations)
        if relative_threshold is not None:
            builder.relativeThreshold(relative_threshold)
        if absolute_threshold is not None:
            builder.absoluteThreshold(absolute_threshold)

        super().__init__(builder.build())


class Adam(JavaObjectWrapper):

    def __init__(self,
                 max_evaluations: Optional[int] = None,
                 alpha: Optional[float] = None,
                 beta1: Optional[float] = None,
                 beta2: Optional[float] = None,
                 epsilon: Optional[float] = None,
                 convergence_checker: Optional[ConvergenceChecker] = None) -> None:

        builder = k.jvm_view().Adam.builder()

        if max_evaluations is not None:
            builder.maxEvaluations(max_evaluations)
        if alpha is not None:
            builder.alpha(alpha)
        if beta1 is not None:
            builder.beta1(beta1)
        if beta2 is not None:
            builder.beta2(beta2)
        if epsilon is not None:
            builder.epsilon(epsilon)
        if convergence_checker is not None:
            builder.convergenceChecker(convergence_checker.unwrap())

        super().__init__(builder.build())


class NonGradientOptimizer(Optimizer):

    def __init__(self, net: Union[BayesNet, Vertex], algorithm: Optional[JavaObjectWrapper] = None) -> None:
        builder, net = Optimizer._build_bayes_net(k.jvm_view().Keanu.Optimizer.NonGradient, net)

        if algorithm is not None:
            builder.algorithm(algorithm.unwrap())

        super(NonGradientOptimizer, self).__init__(builder.build(), net)


class BOBYQA(JavaObjectWrapper):

    def __init__(self,
                 max_evaluations: Optional[int] = None,
                 bounds_range: Optional[float] = None,
                 initial_trust_region_radius: Optional[float] = None,
                 stopping_trust_region_radius: Optional[float] = None) -> None:

        builder = k.jvm_view().BOBYQA.builder()

        if max_evaluations is not None:
            builder.maxEvaluations(max_evaluations)
        if bounds_range is not None:
            builder.boundsRange(bounds_range)
        if initial_trust_region_radius is not None:
            builder.initialTrustRegionRadius(initial_trust_region_radius)
        if stopping_trust_region_radius is not None:
            builder.stoppingTrustRegionRadius(stopping_trust_region_radius)

        super().__init__(builder.build())
