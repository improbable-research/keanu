package io.improbable.keanu.backend;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public interface ProbabilisticGraph extends AutoCloseable {

    default double logProb() {
        return logProb(Collections.emptyMap());
    }

    double logProb(Map<String, ?> inputs);

    LogProbWithSample logProbWithSample(Map<String, ?> inputs, List<String> outputs);

    @Override
    default void close() {
    }
}
