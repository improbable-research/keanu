package io.improbable.keanu.vertices.update;

import io.improbable.keanu.vertices.Vertex;

public class ProbabilisticValueUpdater<T> implements ValueUpdater<T> {
    @Override
    public T updateValue(Vertex<T> vertex) {
        if (!vertex.hasValue()) {
            vertex.setValue(vertex.sample());
        }
        return vertex.getValue();
    }
}
