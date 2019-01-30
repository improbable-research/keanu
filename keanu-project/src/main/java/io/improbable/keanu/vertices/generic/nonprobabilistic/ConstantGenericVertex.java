package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.generic.GenericVertex;

public class ConstantGenericVertex<T> extends GenericVertex<T> implements NonProbabilistic<T>, NonSaveableVertex {

    public ConstantGenericVertex(T value) {
        setValue(value);
    }

    @Override
    public T sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public T calculate() {
        return getValue();
    }
}
