package io.improbable.keanu.vertices.dbltensor.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;

import java.util.Collections;
import java.util.Map;

public class ConstantTensorVertex extends NonProbabilisticDoubleTensor {

    public ConstantTensorVertex(DoubleTensor constant) {
        setValue(constant);
    }

    public ConstantTensorVertex(double constant) {
        this(DoubleTensor.scalar(constant));
    }

    @Override
    protected TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers) {
        return new TensorDualNumber(getValue(), Collections.emptyMap());
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public DoubleTensor getDerivedValue() {
        return getValue();
    }

}
