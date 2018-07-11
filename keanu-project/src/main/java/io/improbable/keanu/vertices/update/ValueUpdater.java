package io.improbable.keanu.vertices.update;

import io.improbable.keanu.vertices.Vertex;

public interface ValueUpdater<T> {

    boolean hasValue(Vertex<T> tVertex);

    T calculateValue(Vertex<T> tVertex);
}
