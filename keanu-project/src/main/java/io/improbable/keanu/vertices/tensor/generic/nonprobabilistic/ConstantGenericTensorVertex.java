package io.improbable.keanu.vertices.tensor.generic.nonprobabilistic;

import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.generic.GenericTensorVertex;

public class ConstantGenericTensorVertex<T> extends VertexImpl<GenericTensor<T>, GenericTensorVertex<T>> implements GenericTensorVertex<T>, NonProbabilistic<GenericTensor<T>>, NonSaveableVertex {

    public ConstantGenericTensorVertex(GenericTensor<T> value) {
        setValue(value);
    }

    public ConstantGenericTensorVertex(T value) {
        setValue(GenericTensor.scalar(value));
    }

    @Override
    public GenericTensor<T> calculate() {
        return getValue();
    }
}