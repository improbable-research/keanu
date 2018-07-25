package io.improbable.keanu.vertices.generic;

import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.update.ValueUpdater;

public abstract class GenericVertex<T> extends Vertex<T> {
    public GenericVertex(ValueUpdater<T> valueUpdater, Observable<T> observation) {
        super(valueUpdater, observation);
    }
}
