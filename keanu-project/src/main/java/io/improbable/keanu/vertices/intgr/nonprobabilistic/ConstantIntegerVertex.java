package io.improbable.keanu.vertices.intgr.nonprobabilistic;


import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class ConstantIntegerVertex extends IntegerVertex {

    public ConstantIntegerVertex(IntegerTensor constant) {
        super(new NonProbabilisticValueUpdater<>(v -> v.getValue()));
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
}
