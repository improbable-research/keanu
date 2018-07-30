package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;

import java.util.Map;

public abstract class NonProbabilisticBool extends BoolVertex {

    @Override
    public double logPmf(BooleanTensor value) {
        return this.getDerivedValue().elementwiseEquals(value).allTrue() ? 0.0 : Double.NEGATIVE_INFINITY;
    }

    @Override
    public Map<Long, DoubleTensor> dLogPmf(BooleanTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isProbabilistic() {
        return false;
    }

    @Override
    public BooleanTensor updateValue() {
        if (!this.isObserved()) {
            setValue(getDerivedValue());
        }
        return getValue();
    }

    public abstract BooleanTensor getDerivedValue();
}
