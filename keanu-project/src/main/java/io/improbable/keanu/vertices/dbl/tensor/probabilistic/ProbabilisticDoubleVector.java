package io.improbable.keanu.vertices.dbl.tensor.probabilistic;

import io.improbable.keanu.vertices.dbl.tensor.DoubleTensor;
import io.improbable.keanu.vertices.dbl.tensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbl.tensor.nonprobabilistic.diff.DualNumber;

public abstract class ProbabilisticDoubleVector extends DoubleTensorVertex {

    @Override
    public DoubleTensor updateValue() {
        return getValue();
    }

    @Override
    public DoubleTensor lazyEval() {
        return getValue();
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

    @Override
    public DualNumber getDualNumber() {
        return new DualNumber(getValue(), getId());
    }
}
