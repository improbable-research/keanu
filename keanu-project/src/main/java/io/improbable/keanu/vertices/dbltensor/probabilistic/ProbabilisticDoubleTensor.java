package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.DualNumber;

import java.util.Map;

public abstract class ProbabilisticDoubleTensor extends DoubleTensorVertex {

    @Override
    public DoubleTensor updateValue() {
        return getValue();
    }

    @Override
    public DoubleTensor lazyEval() {
        if (!hasValue()) {
            setValue(sample());
        }
        return getValue();
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return new DualNumber(getValue(), getId());
    }

}
