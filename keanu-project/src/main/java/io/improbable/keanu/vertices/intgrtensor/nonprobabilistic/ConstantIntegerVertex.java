package io.improbable.keanu.vertices.intgrtensor.nonprobabilistic;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class ConstantIntegerVertex extends NonProbabilisticInteger {

    public ConstantIntegerVertex(IntegerTensor constant) {
        setValue(constant);
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
