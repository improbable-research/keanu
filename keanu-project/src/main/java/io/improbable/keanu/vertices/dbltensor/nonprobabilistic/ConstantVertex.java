package io.improbable.keanu.vertices.dbltensor.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.DualNumber;

import java.util.Collections;
import java.util.Map;

public class ConstantVertex extends NonProbabilisticDoubleTensor {

    public ConstantVertex(DoubleTensor constant) {
        setValue(constant);
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return new DualNumber(getValue(), Collections.emptyMap());
    }

    @Override
    public DoubleTensor sample() {
        return getValue();
    }

    @Override
    public DoubleTensor lazyEval() {
        return getValue();
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return getValue();
    }

}
