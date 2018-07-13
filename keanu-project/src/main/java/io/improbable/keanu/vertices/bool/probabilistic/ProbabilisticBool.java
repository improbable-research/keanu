package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public abstract class ProbabilisticBool extends BoolVertex implements Probabilistic<BooleanTensor> {
    public ProbabilisticBool() {
        super(new ProbabilisticValueUpdater<>());
    }

}
