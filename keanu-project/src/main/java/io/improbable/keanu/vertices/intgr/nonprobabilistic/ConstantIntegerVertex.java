package io.improbable.keanu.vertices.intgr.nonprobabilistic;

public class ConstantIntegerVertex extends NonProbabilisticInteger {

    public ConstantIntegerVertex(Integer constant) {
        setValue(constant);
    }

    @Override
    public Integer sample() {
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
