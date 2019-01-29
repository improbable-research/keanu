package io.improbable.keanu.algorithms;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * A model composed of {@link Variable}s that can return a value of log-probability for a given set of input values.
 */
public interface ProbabilisticModel {

    default double logProb() {
        return logProb(Collections.emptyMap());
    }

    double logProb(Map<VariableReference, ?> inputs);

    double logProbAfter(Map<VariableReference, Object> newValues, double logProbBefore);

    default double logLikelihood() {
        return logLikelihood(Collections.emptyMap());
    }

    double logLikelihood(Map<VariableReference, ?> inputs);

    List<? extends Variable> getLatentVariables();

    List<? extends Variable<DoubleTensor, ?>> getContinuousLatentVariables();
}
