package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.vertices.bool.BoolVertex;

import java.util.Map;

public abstract class NonProbabilisticBool extends BoolVertex {

    @Override
    public double logPmf(Boolean value) {
        return this.getDerivedValue().equals(value) ? 0.0 : Double.NEGATIVE_INFINITY;
    }

    @Override
    public Map<String, Double> dLogPmf(Boolean value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isProbabilistic() {
        return false;
    }

    @Override
    public Boolean updateValue() {
        setValue(getDerivedValue());
        return getValue();
    }

    public abstract Boolean getDerivedValue();
}
