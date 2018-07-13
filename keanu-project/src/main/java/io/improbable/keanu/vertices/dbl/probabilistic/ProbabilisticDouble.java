package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Map;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.update.ProbabilisticValueUpdater;

public abstract class ProbabilisticDouble extends DoubleVertex implements Differentiable {

    public ProbabilisticDouble() {
        super(new ProbabilisticValueUpdater<>());
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        if (isObserved()) {
            return DualNumber.createConstant(getValue());
        } else {
            return DualNumber.createWithRespectToSelf(getId(), getValue());
        }
    }
}
