package io.improbable.keanu.vertices.bool.nonprobabilistic;

public class ConstantBoolVertex extends NonProbabilisticBool {

    public ConstantBoolVertex(Boolean constant) {
        setValue(constant);
    }

    @Override
    public Boolean sample() {
        return getValue();
    }

    @Override
    public Boolean lazyEval() {
        return getValue();
    }

    @Override
    public Boolean getDerivedValue() {
        return getValue();
    }
}
