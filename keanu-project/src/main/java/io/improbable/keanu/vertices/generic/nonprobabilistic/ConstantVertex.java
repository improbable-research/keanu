package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class ConstantVertex<T> extends NonProbabilistic<T> {

    public ConstantVertex(T value) {
        setValue(value);
    }

    @Override
    public T sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public T lazyEval() {
        return getValue();
    }

    @Override
    public T getDerivedValue() {
        return getValue();
    }
}
