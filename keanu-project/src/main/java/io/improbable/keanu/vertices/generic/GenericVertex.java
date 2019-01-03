package io.improbable.keanu.vertices.generic;

import io.improbable.keanu.vertices.Vertex;

public abstract class GenericVertex<T> extends Vertex<T> {

    public GenericVertex() {
        super();
    }

    public GenericVertex(long[] shape) {
        super(shape);
    }
}
