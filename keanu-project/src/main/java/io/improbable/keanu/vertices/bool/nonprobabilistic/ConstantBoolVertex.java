package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class ConstantBoolVertex extends NonProbabilisticBool {

    public static final BoolVertex TRUE = new ConstantBoolVertex(true);
    public static final BoolVertex FALSE = new ConstantBoolVertex(false);

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
