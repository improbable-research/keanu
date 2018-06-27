package io.improbable.keanu.vertices.generic.probabilistic;

import io.improbable.keanu.vertices.Vertex;

public abstract class Probabilistic<T> extends Vertex<T> {

    @Override
    public T updateValue() {
        if (!hasValue()) {
            setValue(sample());
        }
        return getValue();
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

}
