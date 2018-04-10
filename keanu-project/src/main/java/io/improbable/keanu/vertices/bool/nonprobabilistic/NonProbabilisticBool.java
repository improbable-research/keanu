package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.vertices.bool.BoolVertex;

import java.util.Map;

public abstract class NonProbabilisticBool extends BoolVertex {

    @Override
    public double density(Boolean value) {
        return this.getDerivedValue().equals(value) ? 1.0 : 0.0;
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
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
