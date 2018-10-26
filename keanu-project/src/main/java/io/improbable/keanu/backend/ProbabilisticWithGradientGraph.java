package io.improbable.keanu.backend;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

public interface ProbabilisticWithGradientGraph extends ProbabilisticGraph, AutoCloseable {

    Map<String, DoubleTensor> logProbGradients(Map<String, ?> inputs);

    @Override
    default void close() {
    }
}
