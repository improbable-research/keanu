package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import java.util.Collections;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;

public class ConstantDoubleVertex extends NonProbabilisticDouble implements Differentiable {

    public ConstantDoubleVertex(DoubleTensor constant) {
        super(v -> v.getValue());
        setValue(constant);
    }

    public ConstantDoubleVertex(double constant) {
        this(DoubleTensor.scalar(constant));
    }

    public ConstantDoubleVertex(double[] vector) {
        this(DoubleTensor.create(vector));
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        return new DualNumber(getValue(), Collections.emptyMap());
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return getValue();
    }

}
