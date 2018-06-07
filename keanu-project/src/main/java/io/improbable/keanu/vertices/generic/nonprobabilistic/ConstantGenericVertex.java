package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ConstantGenericVertex<T> extends NonProbabilistic<T> {

    public ConstantGenericVertex(T value) {
        setValue(value);
    }

    @Override
    public T sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public T getDerivedValue() {
        return getValue();
    }
}
