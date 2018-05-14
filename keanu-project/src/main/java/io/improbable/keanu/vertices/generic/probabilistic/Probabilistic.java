package io.improbable.keanu.vertices.generic.probabilistic;

import io.improbable.keanu.vertices.Vertex;

import java.util.concurrent.ThreadLocalRandom;

public abstract class Probabilistic<T> extends Vertex<T> {

    @Override
    public T updateValue() {
        return getValue();
    }

    @Override
    public T lazyEval() {
        if (!hasValue()) {
            setValue(sample(ThreadLocalRandom.current()));
        }
        return getValue();
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

}
