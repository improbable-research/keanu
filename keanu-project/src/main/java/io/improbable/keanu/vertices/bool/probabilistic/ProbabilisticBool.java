package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.vertices.bool.BoolVertex;

public abstract class ProbabilisticBool extends BoolVertex {

    @Override
    public Boolean updateValue() {
        return getValue();
    }

    @Override
    public Boolean lazyEval() {
        return getValue();
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

}
