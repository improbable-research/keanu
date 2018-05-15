package io.improbable.keanu.vertices.generic.nonprobabilistic;

public class ConstantVertex<T> extends NonProbabilistic<T> {

    public ConstantVertex(T value) {
        setValue(value);
    }

    @Override
    public T sample() {
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
