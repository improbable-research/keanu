package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.backend.LogProbWithSample;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public interface ProbabilisticGraph extends AutoCloseable {

    default double logProb() {
        return logProb(Collections.emptyMap());
    }

    double logProb(Map<VariableReference, ?> inputs);

    default double logLikelihood() {
        return logLikelihood(Collections.emptyMap());
    }

    double logLikelihood(Map<VariableReference, ?> inputs);

    LogProbWithSample logProbWithSample(Map<String, ?> inputs, List<String> outputs);

    List<? extends Variable> getLatentVariables();

    Map<VariableReference, ?> getLatentVariablesValues();

    @Override
    default void close() {
    }
}
