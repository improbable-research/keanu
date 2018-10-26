package io.improbable.keanu.backend;

import java.util.List;
import java.util.Map;

public interface ProbabilisticGraph extends AutoCloseable {

    boolean USE_TENSOR_FLOW = true;

    double logProb(Map<String, ?> inputs);

    LogProbWithSample logProbWithSample(Map<String, ?> inputs, List<String> outputs);

    @Override
    default void close() {
    }
}
