package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observation;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class ConstantBoolVertex extends BoolVertex {

    public static final BoolVertex TRUE = new ConstantBoolVertex(true);
    public static final BoolVertex FALSE = new ConstantBoolVertex(false);

    public ConstantBoolVertex(BooleanTensor constant) {
        super(
            new NonProbabilisticValueUpdater<>(v -> v.getValue()),
            new Observation<>()
        );
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

}
