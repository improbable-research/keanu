package io.improbable.keanu.vertices.dbltensor.nonprobabilistic;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.DualNumber;

import java.util.Collections;

public class ConstantVertex extends NonProbabilisticDoubleTensor {

    public ConstantVertex(DoubleTensor constant) {
        setValue(constant);
    }

    @Override
    public DualNumber getDualNumber() {
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
