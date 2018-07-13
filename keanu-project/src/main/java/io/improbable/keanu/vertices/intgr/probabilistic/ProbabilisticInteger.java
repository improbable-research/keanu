package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public abstract class ProbabilisticInteger extends IntegerVertex implements Probabilistic<IntegerTensor> {

    public ProbabilisticInteger() {
        super(new ProbabilisticValueUpdater<>());
    }

}
