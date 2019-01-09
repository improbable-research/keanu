package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.List;
import java.util.Map;

public interface ProbabilisticWithGradientGraph extends ProbabilisticGraph {

    Map<? extends VariableReference, DoubleTensor> logProbGradients(List<? extends Variable> inputs);

    Map<? extends VariableReference, DoubleTensor> logProbGradients();

    Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients(List<? extends Variable> inputs);

    Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients();

}
