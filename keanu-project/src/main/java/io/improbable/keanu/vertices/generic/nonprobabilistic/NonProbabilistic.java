package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.Map;

public abstract class NonProbabilistic<TENSOR extends Tensor> extends Vertex<TENSOR> {

    @Override
    public double logProb(TENSOR value) {
        return this.getDerivedValue().elementwiseEquals(value).allTrue() ? 0.0 : Double.NEGATIVE_INFINITY;
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(TENSOR value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isProbabilistic() {
        return false;
    }

    @Override
    public TENSOR updateValue() {
        if (!this.isObserved()) {
            setValue(getDerivedValue());
        }
        return getValue();
    }

    public abstract TENSOR getDerivedValue();

}
