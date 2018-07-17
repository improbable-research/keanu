package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class ConstantBooleanVertex extends BooleanVertex {

    public static final BooleanVertex TRUE = new ConstantBooleanVertex(true);
    public static final BooleanVertex FALSE = new ConstantBooleanVertex(false);

    public ConstantBooleanVertex(BooleanTensor constant) {
        super(
            new NonProbabilisticValueUpdater<>(v -> v.getValue()),
            Observable.observableTypeFor(ConstantBooleanVertex.class)
        );
        setValue(constant);
    }

    public ConstantBooleanVertex(boolean constant) {
        this(BooleanTensor.scalar(constant));
    }

    public ConstantBooleanVertex(boolean[] vector) {
        this(BooleanTensor.create(vector));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return getValue();
    }

}
