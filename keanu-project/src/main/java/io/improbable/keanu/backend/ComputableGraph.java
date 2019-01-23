package io.improbable.keanu.backend;

import java.util.Collection;
import java.util.Map;

import static java.util.Collections.singletonList;

public interface ComputableGraph extends AutoCloseable {

    default <T> T compute(Map<VariableReference, ?> inputs, VariableReference output) {
        return (T) compute(inputs, singletonList(output)).get(output);
    }

    Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs, Collection<VariableReference> outputs);

    <T> T getInput(VariableReference input);

    @Override
    default void close() {
    }
}
