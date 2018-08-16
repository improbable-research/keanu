package io.improbable.keanu.vertices.update;

import java.util.function.Function;

import io.improbable.keanu.vertices.Vertex;

public class NonProbabilisticValueUpdater<T> implements ValueUpdater<T> {

    private Function<Vertex<T>, T> calculation;

    public NonProbabilisticValueUpdater(Function<Vertex<T>, T> calculation) {
        this.calculation = calculation;
    }

    public T updateValue(Vertex<T> vertex) {
        if (!vertex.isObserved()) {
            vertex.setValue(calculation.apply(vertex));
        }
        return vertex.getValue();
    }
}
