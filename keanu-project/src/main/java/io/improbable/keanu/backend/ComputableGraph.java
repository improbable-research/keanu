package io.improbable.keanu.backend;

import io.improbable.keanu.algorithms.VariableReference;

import java.util.Map;

public interface ComputableGraph extends AutoCloseable {

    Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs);

    <T> T getInput(VariableReference input);

    @Override
    default void close() {
    }
}
