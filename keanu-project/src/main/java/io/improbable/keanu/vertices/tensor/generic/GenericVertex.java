package io.improbable.keanu.vertices.tensor.generic;

import io.improbable.keanu.vertices.Vertex;

public interface GenericVertex<T> extends Vertex<T, GenericVertex<T>> {

    @Override
    default Class<?> ofType() {
        return Object.class;
    }
}
