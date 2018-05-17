package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Random;

public class ConstantIntegerVertex extends NonProbabilisticInteger {

    public ConstantIntegerVertex(Integer constant) {
        setValue(constant);
    }

    @Override
    public Integer sample(KeanuRandom random) {
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
