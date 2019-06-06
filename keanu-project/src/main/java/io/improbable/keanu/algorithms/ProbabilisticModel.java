package io.improbable.keanu.algorithms;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * A model composed of {@link Variable}s that can return a value of log-probability for a given set of input values.
 */
public interface ProbabilisticModel extends AutoCloseable {

    default double logProb() {
        return logProb(Collections.emptyMap());
    }

    double logProb(Map<VariableReference, ?> inputs);

    default double logLikelihood() {
        return logLikelihood(Collections.emptyMap());
    }

    double logLikelihood(Map<VariableReference, ?> inputs);

    List<Variable> getLatentVariables();

    default List<? extends Variable<DoubleTensor, ?>> getContinuousLatentVariables() {
        return getLatentVariables().stream()
            .filter(v -> v.getValue() instanceof DoubleTensor)
            .map(v -> (Variable<DoubleTensor, ?>) v)
            .collect(Collectors.toList());
    }

    default double logProbAfter(Map<VariableReference, Object> newValues, double logProbBefore) {
        return logProb(newValues);
    }

    @Override
    default void close() {

    }
}
