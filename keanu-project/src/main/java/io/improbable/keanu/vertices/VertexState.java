package io.improbable.keanu.vertices;

import io.improbable.keanu.network.VariableState;
import lombok.Value;

import java.util.Optional;

@Value
public class VertexState<T> implements VariableState {
    private T value;
    private boolean isObserved;

    public static final <T> VertexState<T> nullState() {
        return new VertexState<>(null, false);
    }

    public Optional<T> getObservedValue() {
        if (isObserved) {
            return Optional.of(value);
        } else {
            return Optional.empty();
        }
    }
}
