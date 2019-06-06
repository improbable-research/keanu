package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

public class ConstantGenericTensorVertex<T> extends GenericTensorVertex<T> implements NonProbabilistic<Tensor<T>>, NonSaveableVertex {

    public ConstantGenericTensorVertex(Tensor<T> value) {
        setValue(value);
    }

    @Override
    public Tensor<T> calculate() {
        return getValue();
    }
}