package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class ConstantGenericVertex<T> extends NonProbabilistic<T> {

    public ConstantGenericVertex(T value) {
        super(v -> v.getValue());
        setValue(value);
    }

    @Override
    public T sample(KeanuRandom random) {
        return getValue();
    }
}
