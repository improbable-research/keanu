package io.improbable.keanu.backend;

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

    List<? extends Variable> getLatentVariables();

    @Override
    default void close() {
    }
}
