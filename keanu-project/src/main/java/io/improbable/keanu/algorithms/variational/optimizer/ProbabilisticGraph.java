package io.improbable.keanu.algorithms.variational.optimizer;

import java.util.Collections;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface ProbabilisticGraph {

    default double logProb() {
        return logProb(Collections.emptyMap());
    }

    double logProb(Map<VariableReference, ?> inputs);

    default double logLikelihood() {
        return logLikelihood(Collections.emptyMap());
    }

    double logLikelihood(Map<VariableReference, ?> inputs);

    List<? extends Variable> getLatentVariables();

    List<? extends Variable<DoubleTensor>> getContinuousLatentVariables();

}
