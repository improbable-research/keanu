package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public interface Observable<T> {
    void observe(T value);
    void unobserve();
    boolean isObserved();

    public static <T> Observable<T> observableTypeFor(Class<? extends Vertex> v) {
        if (!Probabilistic.class.isAssignableFrom(v)) {
            if (IntegerVertex.class.isAssignableFrom(v) || DoubleVertex.class.isAssignableFrom(v)) {
                return new NotObservable<>();
            }
        }
        return new Observation<>();
    }
}
