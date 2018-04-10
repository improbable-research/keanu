package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;

import java.util.Map;

public abstract class NonProbabilistic<T> extends Vertex<T> {

    @Override
    public double density(T value) {
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
    public T updateValue() {
        if (!this.isObserved()) {
            setValue(getDerivedValue());
        }
        return getValue();
    }

    public abstract T getDerivedValue();

}
