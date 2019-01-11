package io.improbable.keanu.vertices;

import io.improbable.keanu.network.VariableState;
import io.improbable.keanu.tensor.Tensor;

import java.util.Objects;
import java.util.Optional;

public class VertexState<T> implements VariableState {


    private final Observable<T> observation;
    private T value;

    public VertexState(Vertex<T> vertex) {
        this(Observable.observableTypeFor(vertex.getClass()),null);
    }

    private VertexState(Observable<T> observation, T value) {
        this.observation = observation;
        this.value = value;
    }

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        if (!observation.isObserved()) {
            this.value = value;
        }
    }

    public boolean hasValue() {
        if (value instanceof Tensor) {
            return !((Tensor) value).isShapePlaceholder();
        } else {
            return value != null;
        }
    }

    public boolean isObserved() {
        return observation.isObserved();
    }

    public void observe(T value) {
        this.value = value;
        observation.observe(value);
    }

    public void unobserve() {
        observation.unobserve();
    }

    public Optional<T> getObservedValue() {
        return observation.getObservedValue();
    }

    public VertexState copy() {
        return new VertexState(observation.copy(), value);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        VertexState<?> that = (VertexState<?>) o;
        return Objects.equals(observation, that.observation) &&
            Objects.equals(value, that.value);
    }

    @Override
    public int hashCode() {

        return Objects.hash(observation, value);
    }
}
