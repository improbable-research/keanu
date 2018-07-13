package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public abstract class ProbabilisticBool extends BoolVertex {
    public ProbabilisticBool() {
        super(new ProbabilisticValueUpdater<>());
    }

}
