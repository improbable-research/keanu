package io.improbable.keanu.vertices;

public interface NonProbabilistic<T> extends Observable<T> {

    default boolean contradictsObservation() {
        return isObserved() && getObservedValue().map(v -> !v.equals(calculate())).orElse(false);
    }

    T calculate();
}
