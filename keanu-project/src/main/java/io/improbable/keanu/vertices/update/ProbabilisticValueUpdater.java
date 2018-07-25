package io.improbable.keanu.vertices.update;

import io.improbable.keanu.vertices.Vertex;

public class ProbabilisticValueUpdater<T> implements ValueUpdater<T> {

    @Override
    public boolean hasValue(Vertex<T> v) {
        return v.hasValue();
    }

    @Override
    public T calculateValue(Vertex<T> v) {
        return v.sample();
    }
}
