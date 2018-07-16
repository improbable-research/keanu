package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.GenericVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class ConstantGenericVertex<T> extends GenericVertex<T> {

    public ConstantGenericVertex(T value) {
        super(
            new NonProbabilisticValueUpdater<>(v -> v.getValue()),
            Observable.observableTypeFor(ConstantGenericVertex.class)
        );
        setValue(value);
    }

    @Override
    public T sample(KeanuRandom random) {
        return getValue();
    }
}
