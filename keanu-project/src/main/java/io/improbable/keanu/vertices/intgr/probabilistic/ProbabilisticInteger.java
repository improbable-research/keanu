package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.concurrent.ThreadLocalRandom;

public abstract class ProbabilisticInteger extends IntegerVertex {

    @Override
    public Integer updateValue() {
        return getValue();
    }

    @Override
    public Integer lazyEval() {
        if (!hasValue()) {
            setValue(sample(ThreadLocalRandom.current()));
        }
        return getValue();
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

}
