package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Map;

public class ConstantDoubleVertex extends NonProbabilisticDouble {

    public ConstantDoubleVertex(Double constant) {
        setValue(constant);
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return DualNumber.createConstant(getValue());
    }

    @Override
    public Double sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public Double lazyEval() {
        return getValue();
    }

    @Override
    public Double getDerivedValue() {
        return getValue();
    }

}
