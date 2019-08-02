package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

public class ConstantGenericTensorVertex<T> extends GenericTensorVertex<T> implements NonProbabilistic<GenericTensor<T>>, NonSaveableVertex {

    public ConstantGenericTensorVertex(GenericTensor<T> value) {
        setValue(value);
    }

    @Override
    public GenericTensor<T> calculate() {
        return getValue();
    }
}