package io.improbable.keanu.vertices.tensor.generic.nonprobabilistic;

import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.generic.GenericVertex;

public class ConstantGenericVertex<T> extends VertexImpl<T, GenericVertex<T>> implements GenericVertex<T>, NonProbabilistic<T>, NonSaveableVertex {

    public ConstantGenericVertex(T value) {
        setValue(value);
    }

    @Override
    public T calculate() {
        return getValue();
    }
}
