package io.improbable.keanu.vertices.generic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.GenericSliceVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.GenericTakeVertex;

public abstract class GenericTensorVertex<T> extends GenericVertex<Tensor<T>> {

    public GenericTensorVertex() {
        super();
    }

    public GenericTensorVertex(long[] shape) {
        super(shape);
    }

    public GenericTakeVertex<T> take(long... index) {
        return new GenericTakeVertex<>(this, index);
    }

    public GenericSliceVertex<T> slice(int dimension, int index) {
        return new GenericSliceVertex<>(this, dimension, index);
    }
}
