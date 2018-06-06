package io.improbable.keanu.vertices.generic.probabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;

public abstract class Probabilistic<T, TENSOR extends Tensor<T>> extends Vertex<TENSOR> {

    @Override
    public TENSOR updateValue() {
        if (!hasValue()) {
            setValue(sampleUsingDefaultRandom());
        }
        return getValue();
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

}
