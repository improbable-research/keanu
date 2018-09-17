package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ConstantGenericVertex<T> extends Vertex<T> implements NonProbabilistic<T> {

    public ConstantGenericVertex(T value) {
        setValue(value);
    }

    @Override
    public T sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public void calculate() {
        setValue(getValue());
    }
}
