package io.improbable.keanu.vertices.booltensor.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.NonProbabilisticBool;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class ConstantBoolVertex extends NonProbabilisticBool {

    public ConstantBoolVertex(BooleanTensor constant) {
        setValue(constant);
    }

    public ConstantBoolVertex(boolean constant) {
        setValue(BooleanTensor.scalar(constant));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public BooleanTensor getDerivedValue() {
        return getValue();
    }
}
