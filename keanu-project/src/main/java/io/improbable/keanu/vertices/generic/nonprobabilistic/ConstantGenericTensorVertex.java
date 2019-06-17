package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.generic.GenericTensorVertex;

public class ConstantGenericTensorVertex<T, TENSOR extends Tensor<T, TENSOR>> extends GenericTensorVertex<T, TENSOR> implements NonProbabilistic<TENSOR>, NonSaveableVertex {

    public ConstantGenericTensorVertex(TENSOR value) {
        setValue(value);
    }

    @Override
    public TENSOR calculate() {
        return getValue();
    }
}