package io.improbable.keanu.vertices.booltensor.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.booltensor.BoolVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class ConstantBoolVertex extends NonProbabilisticBool {

    public static final BoolVertex TRUE = new ConstantBoolVertex(true);
    public static final BoolVertex FALSE = new ConstantBoolVertex(false);

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
