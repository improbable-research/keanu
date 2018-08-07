package io.improbable.keanu.vertices.update;

import java.util.function.Function;

import io.improbable.keanu.vertices.Vertex;

public class NonProbabilisticValueUpdater<T> implements ValueUpdater<T> {

    private Function<Vertex<T>, T> calculation;

    public NonProbabilisticValueUpdater(Function<Vertex<T>, T> calculation) {
        this.calculation = calculation;
    }

    @Override
    public boolean hasValue(Vertex<T> v) {
        return v.isObserved();
    }

    @Override
    public T calculateValue(Vertex<T> v) {
        return calculation.apply(v);
    }
}
