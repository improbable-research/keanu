package io.improbable.keanu.vertices.generic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.GenericSliceVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.operators.unary.GenericTakeVertex;

public abstract class GenericTensorVertex<T, TENSOR extends Tensor<T, TENSOR>> extends GenericVertex<TENSOR> {

    public GenericTensorVertex() {
        super();
    }

    public GenericTensorVertex(long[] shape) {
        super(shape);
    }

    public GenericTakeVertex<T, TENSOR> take(long... index) {
        return new GenericTakeVertex<>(this, index);
    }

    public GenericSliceVertex<T, TENSOR> slice(int dimension, int index) {
        return new GenericSliceVertex<>(this, dimension, index);
    }
}
