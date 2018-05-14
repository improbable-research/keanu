package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.Constant;

import java.util.Random;

public class ConstantVertex<T> extends NonProbabilistic<T> implements Constant<T> {

    public ConstantVertex(T value) {
        setValue(value);
    }

    @Override
    public T sample(Random random) {
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
