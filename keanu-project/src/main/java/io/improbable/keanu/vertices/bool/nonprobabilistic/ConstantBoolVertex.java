package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class ConstantBoolVertex extends NonProbabilisticBool {

    public ConstantBoolVertex(Boolean constant) {
        setValue(constant);
    }

    @Override
    public Boolean sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public Boolean getDerivedValue() {
        return getValue();
    }
}
