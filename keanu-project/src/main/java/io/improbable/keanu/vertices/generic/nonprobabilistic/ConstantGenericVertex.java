package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ConstantGenericVertex<TENSOR extends Tensor> extends NonProbabilistic<TENSOR> {

    public ConstantGenericVertex(TENSOR value) {
        setValue(value);
    }

    @Override
    public TENSOR sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public TENSOR getDerivedValue() {
        return getValue();
    }
}
