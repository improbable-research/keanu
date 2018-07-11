package io.improbable.keanu.vertices.update;

import io.improbable.keanu.vertices.Vertex;

import java.util.function.Function;
import java.util.function.Supplier;

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
