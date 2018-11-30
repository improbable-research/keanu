package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

public interface ProbabilisticWithGradientGraph extends ProbabilisticGraph {

    Map<VariableReference, DoubleTensor> logProbGradients(Map<VariableReference, ?> inputs);

    Map<VariableReference, DoubleTensor> logProbGradients();

    Map<VariableReference, DoubleTensor> logLikelihoodGradients(Map<VariableReference, ?> inputs);

    Map<VariableReference, DoubleTensor> logLikelihoodGradients();

}
