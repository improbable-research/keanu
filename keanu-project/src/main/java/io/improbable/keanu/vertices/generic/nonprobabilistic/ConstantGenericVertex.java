package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class ConstantGenericVertex<T> extends Vertex<T> {

    public ConstantGenericVertex(T value) {
        super(
            new NonProbabilisticValueUpdater<>(v -> v.getValue()));
        setValue(value);
    }

    @Override
    public T sample(KeanuRandom random) {
        return getValue();
    }
}
