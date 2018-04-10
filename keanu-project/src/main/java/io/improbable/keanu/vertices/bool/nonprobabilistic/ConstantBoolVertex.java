package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.vertices.Constant;

public class ConstantBoolVertex extends NonProbabilisticBool implements Constant<Boolean> {

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
