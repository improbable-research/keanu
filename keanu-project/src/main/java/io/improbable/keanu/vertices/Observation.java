package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.Tensor;

public class Observation<T> implements Observable<T> {
    private T value = null;

    @Override
    public void observe(T value) {
        this.value = value;
    }

    @Override
    public void unobserve() {
        this.value = null;
    }

    @Override
    public boolean isObserved() {
        return this.value != null;
    }
}
