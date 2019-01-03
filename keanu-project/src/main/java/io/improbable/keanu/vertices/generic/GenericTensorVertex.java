package io.improbable.keanu.vertices.generic;

import io.improbable.keanu.tensor.Tensor;

public abstract class GenericTensorVertex<T extends Tensor> extends GenericVertex<T> {

    public GenericTensorVertex() {
        super();
    }

    public GenericTensorVertex(long[] shape) {
        super(shape);
    }
}
