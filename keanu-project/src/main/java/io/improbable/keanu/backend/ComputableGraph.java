package io.improbable.keanu.backend;

import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;

import java.util.Collection;
import java.util.Map;

public interface ComputableGraph extends AutoCloseable {

    <T> T compute(Map<VariableReference, ?> inputs, VariableReference output);

    Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs, Collection<VariableReference> outputs);

    <T> T getInput(VariableReference input);

    @Override
    default void close() {
    }
}
