package io.improbable.keanu.vertices;

public class Observation<T> implements Observable<T> {
    // package private
    Observation() {}

    private T observedValue = null;

    @Override
    public void observe(T value) {
        this.observedValue = value;
    }

    @Override
    public void unobserve() {
        this.observedValue = null;
    }

    @Override
    public boolean isObserved() {
        return this.observedValue != null;
    }
}
