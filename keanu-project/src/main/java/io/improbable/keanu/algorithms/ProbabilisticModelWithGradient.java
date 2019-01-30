package io.improbable.keanu.algorithms;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

/**
 * A {@link ProbabilisticModel} that can also return the gradient of its log-probability: required by algorithms such as {@link io.improbable.keanu.Keanu.Sampling.NUTS} and {@link io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer}.
 */
public interface ProbabilisticModelWithGradient extends ProbabilisticModel {

    Map<? extends VariableReference, DoubleTensor> logProbGradients(Map<VariableReference, ?> inputs);

    Map<? extends VariableReference, DoubleTensor> logProbGradients();

    Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients(Map<VariableReference, ?> inputs);

    Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients();

}
