package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class ConstantBoolVertex extends BoolVertex implements NonProbabilistic<BooleanTensor> {

    public static final BoolVertex TRUE = new ConstantBoolVertex(true);
    public static final BoolVertex FALSE = new ConstantBoolVertex(false);

    public ConstantBoolVertex(BooleanTensor constant) {
        setValue(constant);
    }

    public ConstantBoolVertex(boolean constant) {
        this(BooleanTensor.scalar(constant));
    }

    public ConstantBoolVertex(boolean[] vector) {
        this(BooleanTensor.create(vector));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public void calculate() {
    }
}
