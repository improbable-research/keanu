package io.improbable.keanu.vertices;

public interface Observable<T> {
    void observe(T value);
    void unobserve();
    boolean isObserved();
}
