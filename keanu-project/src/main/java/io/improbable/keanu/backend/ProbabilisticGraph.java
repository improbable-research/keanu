package io.improbable.keanu.backend;

import java.util.List;
import java.util.Map;

public interface ProbabilisticGraph extends AutoCloseable {

    double logProb(Map<String, ?> inputs);

    LogProbWithSample logProbWithSample(Map<String, ?> inputs, List<String> outputs);

    @Override
    void close();
}
