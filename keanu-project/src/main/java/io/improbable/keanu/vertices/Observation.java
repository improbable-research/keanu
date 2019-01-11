package io.improbable.keanu.vertices;

import java.util.Optional;

public class Observation<T> implements Observable<T> {
    // package private - because it's created by the factory method Observable.observableTypeFor
    Observation() {
    }

    private T observedValue = null;

    private Observation(T observedValue) {
        this.observedValue = observedValue;
    }

    @Override
    public void observe(T value) {
        this.observedValue = value;
    }

    @Override
    public void unobserve() {
        this.observedValue = null;
    }

    @Override
    public Optional<T> getObservedValue() {
        return Optional.of(observedValue);
    }

    @Override
    public boolean isObserved() {
        return this.observedValue != null;
    }

    @Override
    public Observable copy() {
        return new Observation(observedValue);
    }
}
