package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.vertices.Constant;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Collections;
import java.util.Map;

public class ConstantDoubleVertex extends NonProbabilisticDouble implements Constant<Double> {

    public ConstantDoubleVertex(Double constant) {
        setValue(constant);
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return new DualNumber(getValue(), Collections.emptyMap());
    }

    @Override
    public Double sample() {
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
