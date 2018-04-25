package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.vertices.intgr.IntegerVertex;

public abstract class ProbabilisticInteger extends IntegerVertex {

    @Override
    public Integer updateValue() {
        return getValue();
    }

    @Override
    public Integer lazyEval() {
        if (!hasValue()) {
            setValue(sample());
        }
        return getValue();
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

}
