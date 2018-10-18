package io.improbable.keanu.backend;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface ProbabilisticWithGradientGraph extends ProbabilisticGraph, AutoCloseable {

    Map<String, DoubleTensor> logProbGradients(Map<String, ?> inputs);

    @Override
    void close();
}
