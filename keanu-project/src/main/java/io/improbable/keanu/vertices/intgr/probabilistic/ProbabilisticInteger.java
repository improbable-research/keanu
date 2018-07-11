package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;
import io.improbable.keanu.vertices.update.ValueUpdater;

public abstract class ProbabilisticInteger extends IntegerVertex {

    public ProbabilisticInteger() {
        super(new ProbabilisticValueUpdater<>());
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

}
