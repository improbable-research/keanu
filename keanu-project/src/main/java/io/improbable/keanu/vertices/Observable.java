package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.Optional;

public interface Observable<T> {
    void observe(T value);

    void unobserve();

    Optional<T> getObservedValue();

    boolean isObserved();

    static boolean isObservable(Class<? extends Vertex> v) {
        boolean isProbabilistic = Probabilistic.class.isAssignableFrom(v);
        boolean isNotDoubleOrIntegerVertex = !IntegerVertex.class.isAssignableFrom(v) && !DoubleVertex.class.isAssignableFrom(v);

        return isProbabilistic || isNotDoubleOrIntegerVertex;
    }

}
