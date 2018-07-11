package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.generic.probabilistic.Probabilistic;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public abstract class ProbabilisticBool extends BoolVertex {
    public ProbabilisticBool() {
        super(new ProbabilisticValueUpdater<>());
    }

    @Override
    public boolean isProbabilistic() {
        return true;
    }

}
