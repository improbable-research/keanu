package io.improbable.keanu.vertices;

public interface NonProbabilistic<T> extends Observable<T> {
    boolean contradictsObservation();
}
