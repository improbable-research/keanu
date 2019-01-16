package io.improbable.keanu.vertices;

import java.util.Optional;

public interface Observable<T> {
    void observe(T value);

    void unobserve();

    Optional<T> getObservedValue();

    boolean isObserved();
}
