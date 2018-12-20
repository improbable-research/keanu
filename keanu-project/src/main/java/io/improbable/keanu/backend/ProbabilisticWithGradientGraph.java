package io.improbable.keanu.backend;

import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

public interface ProbabilisticWithGradientGraph extends ProbabilisticGraph, AutoCloseable {

    Map<VariableReference, DoubleTensor> logProbGradients(Map<VariableReference, ?> inputs);

    @Override
    default void close() {
    }
}
