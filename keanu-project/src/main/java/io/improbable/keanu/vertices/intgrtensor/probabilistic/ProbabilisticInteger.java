package io.improbable.keanu.vertices.intgrtensor.probabilistic;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgrtensor.IntegerVertex;

public abstract class ProbabilisticInteger extends IntegerVertex {

    @Override
    public IntegerTensor updateValue() {
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
