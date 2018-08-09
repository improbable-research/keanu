package io.improbable.keanu.vertices.update;

import io.improbable.keanu.vertices.Vertex;

public interface ValueUpdater<T> {
    T updateValue(Vertex<T> tVertex);
}
