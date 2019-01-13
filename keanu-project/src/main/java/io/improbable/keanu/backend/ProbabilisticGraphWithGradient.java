package io.improbable.keanu.backend;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

public interface ProbabilisticGraphWithGradient extends ProbabilisticGraph, AutoCloseable {

    Map<? extends VariableReference, DoubleTensor> logProbGradients(Map<VariableReference, ?> inputs);

    Map<? extends VariableReference, DoubleTensor> logProbGradients();

    Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients(Map<VariableReference, ?> inputs);

    Map<? extends VariableReference, DoubleTensor> logLikelihoodGradients();

    @Override
    default void close() {
    }
}
