package io.improbable.keanu.vertices.booltensor.probabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.booltensor.BoolVertex;

public abstract class ProbabilisticBool extends BoolVertex {

    @Override
    public BooleanTensor updateValue() {
        if (!hasValue()) {
            setValue(sampleUsingDefaultRandom());
        }
        return getValue();
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

}
