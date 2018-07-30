package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ConstantIntegerVertex extends NonProbabilisticInteger {

    public ConstantIntegerVertex(IntegerTensor constant) {
        setValue(constant);
    }

    public ConstantIntegerVertex(int[] vector) {
        this(IntegerTensor.create(vector));
    }

    public ConstantIntegerVertex(int constant) {
        this(IntegerTensor.scalar(constant));
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public IntegerTensor getDerivedValue() {
        return getValue();
    }
}
