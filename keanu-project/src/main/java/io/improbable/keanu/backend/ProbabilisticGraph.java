package io.improbable.keanu.backend;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface ProbabilisticGraph extends AutoCloseable {

    double logProb(Map<String, ?> inputs);

    @Override
    void close();
}
