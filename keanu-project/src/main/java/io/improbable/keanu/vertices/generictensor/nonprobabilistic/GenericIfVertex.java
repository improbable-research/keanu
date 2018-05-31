package io.improbable.keanu.vertices.generictensor.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;

public class GenericIfVertex<T> extends IfVertex<T, Tensor<T>> {

    public GenericIfVertex(int[] shape,
                           Vertex<? extends BooleanTensor> predicate,
                           Vertex<? extends Tensor<T>> thn,
                           Vertex<? extends Tensor<T>> els) {
        super(Tensor.placeHolder(shape), predicate, thn, els);
    }

    @Override
    protected Tensor<T> op(BooleanTensor predicate, Tensor<T> thn, Tensor<T> els) {
        return predicate.setIf(thn, els);
    }
}
