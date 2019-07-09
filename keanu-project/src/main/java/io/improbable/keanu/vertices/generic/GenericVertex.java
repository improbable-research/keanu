package io.improbable.keanu.vertices.generic;

import io.improbable.keanu.vertices.VertexImpl;

public abstract class GenericVertex<T> extends VertexImpl<T> {

    public GenericVertex() {
        super();
    }

    public GenericVertex(long[] shape) {
        super(shape);
    }
}
