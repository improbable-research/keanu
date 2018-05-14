package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import java.util.Random;

public class ConstantIntegerVertex extends NonProbabilisticInteger {

    public ConstantIntegerVertex(Integer constant) {
        setValue(constant);
    }

    @Override
    public Integer sample(Random random) {
        return getValue();
    }

    @Override
    public Integer lazyEval() {
        return getValue();
    }

    @Override
    public Integer getDerivedValue() {
        return getValue();
    }
}
