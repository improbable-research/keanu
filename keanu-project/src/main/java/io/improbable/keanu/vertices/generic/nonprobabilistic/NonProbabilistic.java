package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.Map;

public abstract class NonProbabilistic<T> extends Vertex<T> {

    @Override
    public double logProb(T value) {
        return this.getDerivedValue().equals(value) ? 0.0 : Double.NEGATIVE_INFINITY;
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(T value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isProbabilistic() {
        return false;
    }

    @Override
    public T updateValue() {
        if (!this.isObserved()) {
            setValue(getDerivedValue());
        }
        return getValue();
    }

    public abstract T getDerivedValue();

}
